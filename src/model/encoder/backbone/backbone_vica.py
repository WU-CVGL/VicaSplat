from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union, Dict
from jaxtyping import Float

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True # for gpu >= Ampere and pytorch >= 1.12
from functools import partial

from einops import rearrange, repeat

from diffusers.models import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .croco.blocks import DecoderBlock, Block, PatchEmbed, DropPath, Mlp
from .croco.patch_embed import get_patch_embed
from .croco.pos_embed import RoPE2D
from .backbone import Backbone
from ....geometry.camera_emb import get_intrinsic_embedding
from ....misc.rope_utils import apply_rotary_emb, get_rotary_pos_embed


inf = float('inf')

croco_params = {
    'ViTLarge_BaseDecoder': {
        'enc_depth': 24,
        'dec_depth': 12,
        'enc_embed_dim': 1024,
        'dec_embed_dim': 768,
        'enc_num_heads': 16,
        'dec_num_heads': 12,
        'pos_embed': 'RoPE100',
        'img_size': (512, 512),
    },
}

default_dust3r_params = {
    'enc_depth': 24,
    'dec_depth': 12,
    'enc_embed_dim': 1024,
    'dec_embed_dim': 768,
    'enc_num_heads': 16,
    'dec_num_heads': 12,
    'pos_embed': 'RoPE100',
    'patch_embed_cls': 'PatchEmbedDust3R',
    'img_size': (512, 512),
    'head_type': 'dpt',
    'output_mode': 'pts3d',
    'depth_mode': ('exp', -inf, inf),
    'conf_mode': ('exp', 1, inf)
}

class VideoCameraAttention(nn.Module):
    
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        rope_img_2d=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope_img_2d = rope_img_2d
        
    def forward(
        self, 
        img: torch.Tensor,
        cam: torch.Tensor,
        camera_causal_attention_mask: Optional[torch.Tensor] = None,
        freqs_cis_img: tuple[torch.Tensor, torch.Tensor] = None,
        freqs_cis_cam: tuple[torch.Tensor, torch.Tensor] = None,
        img_pos=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert img.ndim == 4
        B, T, N, C = img.shape
        head_dim = C // self.num_heads

        # visual tokens
        qkv_img = self.qkv(img)
        qkv_img = qkv_img.reshape(B, T*N, 3, self.num_heads, head_dim).transpose(1, 3)
        q_img, k_img, v_img = [qkv_img[:,:,i] for i in range(3)]
        
        if self.rope_img_2d is not None:
            img_pos = img_pos.reshape(B, T*N, -1)
            q_img = self.rope_img_2d(q_img, img_pos)
            k_img = self.rope_img_2d(k_img, img_pos)
        else:
            assert freqs_cis_img is not None
            q_img, k_img = apply_rotary_emb(q_img, k_img, freqs_cis_img, head_first=True)
        
        # camera tokens
        qkv_cam = self.qkv(cam)
        qkv_cam = qkv_cam.reshape(B, T, 3, self.num_heads, head_dim).transpose(1, 3)
        q_cam, k_cam, v_cam = [qkv_cam[:,:,i] for i in range(3)]
        if freqs_cis_cam is not None:
            q_cam, k_cam = apply_rotary_emb(q_cam, k_cam, freqs_cis_cam, head_first=True)
        
        # fused K & V
        k_img = k_img.reshape(B, self.num_heads, T, N, head_dim)
        v_img = v_img.reshape(B, self.num_heads, T, N, head_dim)
        k_fused = torch.cat([k_cam.unsqueeze(-2), k_img], dim=-2).reshape(B, self.num_heads, -1, head_dim)
        v_fused = torch.cat([v_cam.unsqueeze(-2), v_img], dim=-2).reshape(B, self.num_heads, -1, head_dim)
        
        # Apply full attention to visual tokens
        x_img = F.scaled_dot_product_attention(
            q_img, k_fused, v_fused).transpose(1, 2).reshape(B, T, N, C)
        # Apply attention to camera tokens
        x_cam = F.scaled_dot_product_attention(
            q_cam, k_fused, v_fused, attn_mask=camera_causal_attention_mask
        ).transpose(1, 2).reshape(B, T, C)
        
        # output
        img_out = self.proj_drop(self.proj(x_img))
        cam_out = self.proj_drop(self.proj(x_cam))
        return img_out, cam_out
    
    
class CrossNeighborAttention(nn.Module):
    
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        rope_img_2d=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        
        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.rope_img_2d = rope_img_2d
        
    def forward(
        self,
        img: torch.Tensor,
        freqs_cis_img: tuple[torch.Tensor, torch.Tensor] = None,
        img_pos=None,
    ) -> torch.Tensor:
        B, T, N, C = img.shape
        head_dim = C // self.num_heads
        
        q = self.projq(img).reshape(B, T*N, self.num_heads, head_dim).transpose(1, 2)
        k = self.projk(img).reshape(B, T*N, self.num_heads, head_dim).transpose(1, 2)
        v = self.projv(img).reshape(B, T*N, self.num_heads, head_dim).transpose(1, 2)
        
        if self.rope_img_2d is not None:
            img_pos = img_pos.reshape(B, T*N, -1)
            q = self.rope_img_2d(q, img_pos)
            k = self.rope_img_2d(k, img_pos)
        else:
            assert freqs_cis_img is not None
            q, k = apply_rotary_emb(q, k, freqs_cis_img, head_first=True)
            
        kv = torch.cat([k, v], dim=0)
        kv = rearrange(kv, "B H (T N) C -> B H T N C", T=T)
        if T == 2:
            kv_ca = torch.roll(kv, shifts=1, dims=2)
        elif T > 2:
            kv_prev, kv_next = torch.roll(kv, 1, dims=2), torch.roll(kv, -1, dims=2)
            kv_prev[:, :, :1] = kv_next[:, :, :1]
            kv_next[:, :, -1:] = kv_prev[:, :, -1:]
            kv_ca = torch.cat([kv_prev, kv_next], dim=3)
        else:
            raise ValueError("Invalid number of frames")
        
        k_ca, v_ca = rearrange(kv_ca, "(K B) H T N C -> K (B T) H N C", K=2)
        q = rearrange(q, "B H (T N) C -> (B T) H N C", T=T)
        
        x = F.scaled_dot_product_attention(q, k_ca, v_ca)
        x = rearrange(x, "(B T) H N C -> B T N (H C)", T=T)
        img_out = self.proj_drop(self.proj(x)) 
        return img_out   


class AdaLNModulation(nn.Module):
    def __init__(
        self, 
        n_channels: int, 
        n_mods: int = 3,
    ):
        super().__init__()
        self.n_mods = n_mods
        self.nonlinear = nn.SiLU()
        self.proj = nn.Linear(
            n_channels, n_mods*n_channels, bias=True,
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, emb: torch.Tensor):
        emb = self.nonlinear(emb)
        return self.proj(emb).chunk(self.n_mods, dim=-1)
    

class MixDecoderBlock(nn.Module):
    
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4., 
        qkv_bias=False, 
        drop=0., 
        attn_drop=0.,
        drop_path=0., 
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, 
        framewise_modulation=True,
        cross_neighbor_attention=True,
        rope_img_2d=None
    ):
        super().__init__()
        self.cam_norm1 = norm_layer(dim)
        if framewise_modulation:
            self.modulation1 = AdaLNModulation(dim, n_mods=3)
        else:
            self.modulation1 = None
            
        self.norm1 = norm_layer(dim)
        self.attn = VideoCameraAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, rope_img_2d=rope_img_2d
        )
        
        self.cam_norm2 = norm_layer(dim)
        if framewise_modulation:
            self.modulation2 = AdaLNModulation(
                dim, n_mods=6 if cross_neighbor_attention else 3
            )
        else:
            self.modulation2 = None
            
        if cross_neighbor_attention:
            self.norm2 = norm_layer(dim)
            self.cross_attn = CrossNeighborAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, rope_img_2d=rope_img_2d
            )
        else:
            self.norm2 = None
            self.cross_attn = None
            
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()   # dummy
        
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_cam = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def _modulate(self, x, scale=None, shift=None):
        if scale is not None:
            x = x * (1 + scale)
        if shift is not None:
            x = x + shift
        return x
    
    def _gate(self, residual, gate=None):
        if gate is not None:
            return (1 + gate) * residual
        return residual
    
    def forward(
        self,
        img: torch.Tensor,
        cam: torch.Tensor,
        camera_causal_attention_mask: Optional[torch.Tensor] = None,
        freqs_cis_img: tuple[torch.Tensor, torch.Tensor] = None,
        freqs_cis_cam: tuple[torch.Tensor, torch.Tensor] = None,
        img_pos=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cam_normed = self.cam_norm1(cam)
        if self.modulation1 is not None:
            scale_sa, shift_sa, gate_sa = self.modulation1(cam_normed.unsqueeze(2))
        else:
            scale_sa = shift_sa = gate_sa = 0.0
            
        img_normed = self.norm1(img)
        img_normed = self._modulate(img_normed, scale_sa, shift_sa)
        
        # self-attn
        img_attn_out, cam_attn_out = self.attn(
            img_normed, cam_normed, camera_causal_attention_mask, freqs_cis_img, freqs_cis_cam, img_pos
        )
        img = img + self._gate(img_attn_out, gate_sa)
        cam = cam + cam_attn_out
        
        # cross-attn
        cam_normed = self.cam_norm2(cam)
        if self.modulation2 is not None:
            if self.cross_attn is not None:
                (
                    scale_ca, shift_ca, gate_ca, scale_mlp, shift_mlp, gate_mlp
                ) = self.modulation2(cam_normed.unsqueeze(2))
            else:
                scale_ca = shift_ca = gate_ca = 0.0
                scale_mlp, shift_mlp, gate_mlp = self.modulation2(cam_normed.unsqueeze(2))
        else:
            scale_ca = shift_ca = gate_ca = scale_mlp = shift_mlp = gate_mlp = 0.0
            
        if self.cross_attn is not None:
            img_normed = self.norm2(img)
            img_normed = self._modulate(img_normed, scale_ca, shift_ca)
            img_attn_out = self.cross_attn(
                img_normed, freqs_cis_img, img_pos
            )
            img = img + self._gate(img_attn_out, gate_ca)
            
        # mlp
        img_normed = self.norm3(img)
        img_normed = self._modulate(img_normed, scale_mlp, shift_mlp)
        img_mlp_out = self.mlp(img_normed)
        img = img + self._gate(img_mlp_out, gate_mlp)
        
        cam_mlp_out = self.mlp_cam(cam_normed)
        cam = cam + cam_mlp_out
        
        return img, cam
            
        


class VicaNet(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    intrinsics_embed_loc: str = 'encoder'
    intrinsics_embed_degree: int = 4
    intrinsics_embed_type: str = 'token'

    @register_to_config
    def __init__(
        self,
        img_size: int | tuple[int] = 224,
        patch_size: int | tuple[int] = 16,
        enc_embed_dim: int = 1024,
        enc_depth: int = 24,
        enc_num_heads: int = 16,
        dec_embed_dim: int = 768,
        dec_depth: int = 12,
        dec_num_heads: int = 12,
        mlp_ratio: float = 4.0,
        temporal_rope_theta: int = 100,
        rope_dim_list: list[int] = [16, 56, 56],
        use_blocked_causal_attention: bool = True,
        use_framewise_modulation: bool = True,
        use_cross_neighbor_attention: bool = True,
        use_intrinsic_embedding: bool = True,
    ):
        # intrinsics embedding
        
        
        super().__init__()
        # layernorm
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        # patch embeddings
        self._set_patch_embed(img_size, patch_size, enc_embed_dim)
        
        if RoPE2D is not None and len(rope_dim_list) == 2:
            self.rope_img_2d = RoPE2D(freq=100)
        else:
            self.rope_img_2d = None
        
        # transformer for the encoder
        self.enc_depth = enc_depth
        self.enc_embed_dim = enc_embed_dim
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer, rope=self.rope_img_2d)
            for i in range(enc_depth)])
        self.enc_norm = self.norm_layer(enc_embed_dim)
        
        # decoder
        self._set_decoder()

        # intrinsic embedding
        if use_intrinsic_embedding:
            self.intrinsic_encoder = nn.Linear(9, 1024)
        else:
            self.intrinsic_encoder = None
        
        # learnable camera token
        self.camera_extrinsic_token = nn.Parameter(
            torch.empty(dec_embed_dim, dtype=torch.float).requires_grad_(True)
        )
        self.camera_intrinsic_token = nn.Parameter(
            torch.empty(dec_embed_dim, dtype=torch.float).requires_grad_(True)
        )
        
        # initializer weights
        self.initialize_weights()
        
        self.gradient_checkpointing = False
        
    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        
    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)
        
    def _set_decoder(self):
        self.decoder_embed = nn.Linear(self.config.enc_embed_dim, self.config.dec_embed_dim, bias=True)
        
        self.dec_blocks = nn.ModuleList(
            [
                MixDecoderBlock(
                    self.config.dec_embed_dim, self.config.dec_num_heads, mlp_ratio=self.config.mlp_ratio, qkv_bias=True,
                    norm_layer=self.norm_layer, framewise_modulation=self.config.use_framewise_modulation,
                    cross_neighbor_attention=self.config.use_cross_neighbor_attention, rope_img_2d=self.rope_img_2d
                ) for _ in range(self.config.dec_depth)
            ]
        )
        self.dec_norm = self.norm_layer(self.config.dec_embed_dim)
        self.camera_dec_norm = self.norm_layer(self.config.dec_embed_dim)
        
    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        # camera tokens
        nn.init.normal_(self.camera_extrinsic_token, std=0.02)
        nn.init.normal_(self.camera_intrinsic_token, std=0.02)
        # linears and layer norms
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _encode_image(self, image, intrins_embed=None):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image)

        # maybe append intrinsic embed
        if intrins_embed is not None:
            x = torch.cat([x, intrins_embed], dim=1)
            add_pos = pos[:, 0:1, :].clone()
            add_pos[:, :, 0] += (pos[:, -1, 0].unsqueeze(-1) + 1)
            pos = torch.cat((pos, add_pos), dim=1)

        # add positional embedding without cls token
        # assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            for blk in self.enc_blocks:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(blk), x, pos, use_reentrant=False
                )
        else:
            for blk in self.enc_blocks:
                x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos
    
    def _decoder(self, x, xpos, freqs_cis_img=None, freqs_cis_cam=None):
        B, T, N, C = x.shape
        x_intermediates = [x]   # before projection
        
        x = self.decoder_embed(x)
        
        # prepare camera tokens and attention mask
        # if self.config.use_intrinsic_embedding:
        #     cam = self.camera_extrinsic_token.expand_as(x[:, :, 0, :])  # (B T L)
        # else:
        cam_extrins_tokens = self.camera_extrinsic_token.expand_as(x[:, 1:, 0, :])  # (B T-1 L)
        cam_intrins_tokens = self.camera_intrinsic_token.expand_as(x[:, :1, 0, :])  # (B 1 L)
        cam = torch.cat([cam_intrins_tokens, cam_intrins_tokens+cam_extrins_tokens], dim=1) # (B T L)
        
        if self.config.use_blocked_causal_attention:
            cam_attn_mask = self._produce_camera_blocked_causal_attention_mask(
                T, N, first_token_full_attn=not self.config.use_intrinsic_embedding
            ).to(x.device)
        else:
            cam_attn_mask = None
        
        ## Transformer forward
        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            for blk in self.dec_blocks:
                x, cam = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(blk),
                    x, cam, cam_attn_mask, freqs_cis_img, freqs_cis_cam, xpos,
                    use_reentrant=False
                )
                x_intermediates.append(x)
        else:
            for blk in self.dec_blocks:
                x, cam = blk(x, cam, cam_attn_mask, freqs_cis_img, freqs_cis_cam, xpos)
                x_intermediates.append(x)
            
        x_intermediates[-1] = self.dec_norm(x_intermediates[-1])
        cam = self.camera_dec_norm(cam)
        return x_intermediates, cam
    
    def forward(
        self,
        x: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        B, C, T, H, W = x.shape

        if self.config.use_intrinsic_embedding:
            assert intrinsics is not None
            intrinsic_embedding = self.intrinsic_encoder(intrinsics.flatten(2)) # (B, T, L)
            intrinsic_embedding = rearrange(intrinsic_embedding, "B T C -> (B T) () C")
        else:
            intrinsic_embedding = None

        # encode image
        x, xpos = self._encode_image(
            rearrange(x, "B C T H W -> (B T) C H W"), intrinsic_embedding
        )
        x = rearrange(x, "(B T) N C -> B T N C", T=T)
        xpos = rearrange(xpos, "(B T) N K -> B T N K", T=T)
        
        # prepare rope freqs
        if self.rope_img_2d is None:
            h = H // self.config.patch_size
            w = W // self.config.patch_size
            if len(self.config.rope_dim_list) == 3:
                freqs_cis_img = self._prepare_rope_freqs(
                    (T, h, w), n_dim=3, rope_dim_list=self.config.rope_dim_list, rope_theta=[self.config.temporal_rope_theta, 100, 100]
                )
            elif len(self.config.rope_dim_list) == 2:
                # assert self.backbone.config.use_framewise_modulation
                freqs_cis_img = self._prepare_rope_freqs(
                    (h, w), n_dim=2, rope_dim_list=self.config.rope_dim_list, rope_theta=100
                )
            else:
                raise ValueError
        else:
            freqs_cis_img = None
            
        freqs_cis_camera = self._prepare_rope_freqs((T,), n_dim=1, rope_theta=self.config.temporal_rope_theta)
            
        # decoder
        x_intermediates, camera = self._decoder(x, xpos, freqs_cis_img, freqs_cis_camera)
        
        if intrinsic_embedding is not None:
            for i in range(len(x_intermediates)):
                x_intermediates[i] = x_intermediates[i][:, :, :-1]
        
        x_final = x_intermediates[-1]

        if self.config.use_intrinsic_embedding:
            camera_intrinsic = None
            camera_extrinsic = camera[:, 1:]    # drop the first one, which is always identity pose
        else:
            camera_intrinsic, camera_extrinsic = camera[:, 0], camera[:, 1:]
        
        return x_final, camera_extrinsic, camera_intrinsic, x_intermediates
    
        
    def _produce_camera_blocked_causal_attention_mask(
        self, n_frames, n_visual_tokens_per_frame, first_token_full_attn=True
    ):
        mask = torch.ones(n_frames, n_frames, dtype=torch.bool).tril(diagonal=0)
        if first_token_full_attn:
            mask[:1] = torch.ones_like(mask[:1])    # make the first camera token, which is the intrinsic token, share global scope
        mask = mask[..., None, None].repeat(1, 1, 1, 1+n_visual_tokens_per_frame)
        mask = rearrange(mask, "h w p q -> (h p) (w q)")
        return mask
    
    def _prepare_rope_freqs(self, size: tuple, n_dim=None, rope_theta=100, rope_dim_list=None):
        if n_dim == None:
            n_dim = len(size)
        return get_rotary_pos_embed(
            tensor_size=size,
            patch_size=[1]*n_dim,
            head_dim=self.config.dec_embed_dim // self.config.dec_num_heads,
            rope_theta=rope_theta,
            rope_dim_list=rope_dim_list,
            target_ndim=n_dim,
        )