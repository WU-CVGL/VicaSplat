from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import math
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from omegaconf import DictConfig

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, MyGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from .backbone.backbone_vica import VicaNet
from ...misc.cam_utils import camera_matrix_from_dq_array, camera_matrix_from_qt_array, simple_intrin_matrix_from_fov


inf = float('inf')

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int

@dataclass
class VicaSplatCfg:
    name: str
    backbone: dict | DictConfig
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    predict_opacity: bool
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    gs_center_head_type: str = "dpt"
    gs_param_head_type: str = "dpt_gs"
    predict_conf: bool = False
    camera_type: Literal["dq", "qt"] = "dq"

class VicaSplat(Encoder[VicaSplatCfg]):
    backbone: VicaNet
    gaussian_adapter: GaussianAdapter
    patch_size: int = 16

    def __init__(
        self, 
        cfg: VicaSplatCfg, 
        weight_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ) -> None:
        self.factory_kwargs = {"device": device, "dtype": weight_dtype}

        super().__init__(cfg)
        self.camera_extrinsic_channels = 8 if self.cfg.camera_type == "dq" else 7

        self.backbone = VicaNet(**cfg.backbone)
        self.gaussian_adapter = MyGaussianAdapter(cfg.gaussian_adapter)

        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity
        self.set_center_head(output_mode='pts3d', head_type=cfg.gs_center_head_type, landscape_only=True,
                        depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf) if cfg.predict_conf else None,)
        self.set_gs_params_head(cfg, head_type=cfg.gs_param_head_type)

        self.set_camera_extrinsic_head()
        if not self.backbone.config.use_intrinsic_embedding:
            self.set_camera_intrinsic_head()
        else:
            self.camera_intrinsic_head = None

    
    # borrowed from noposplat
    def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        self.predict_confidence = bool(conf_mode)
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self.backbone, has_conf=self.predict_confidence)
        self.downstream_head1.to(**self.factory_kwargs)

        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
    
    # borrowed from noposplat
    def set_gs_params_head(self, cfg, head_type):
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    self.patch_size ** 2 * self.raw_gs_dim,
                ),
            )

        elif head_type == 'dpt':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)  # for view1 3DGS

        elif head_type == 'dpt_gs':
            self.gaussian_param_head = head_factory(head_type, 'gs_params', self.backbone, has_conf=False, out_nchan=self.raw_gs_dim)
        else:
            raise NotImplementedError(f"unexpected {head_type=}")
        
        self.gaussian_param_head.to(**self.factory_kwargs)
        
    def set_camera_extrinsic_head(self):
        self.camera_extrinsic_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                self.backbone.config.dec_embed_dim, self.camera_extrinsic_channels, **self.factory_kwargs
            )
        )
        # init the head to predict identical camera pose
        nn.init.zeros_(self.camera_extrinsic_head[1].weight)
        nn.init.zeros_(self.camera_extrinsic_head[1].bias)
    
    def set_camera_intrinsic_head(self):
        self.camera_intrinsic_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                self.backbone.config.dec_embed_dim, 2, **self.factory_kwargs
            )
        )
        # init the head to predict fovs as degree 50
        nn.init.zeros_(self.camera_intrinsic_head[1].weight)
        nn.init.constant_(self.camera_intrinsic_head[1].bias, math.pi * 50 / 180)
        
    def enable_gradient_checkpointing(self):
        self.backbone.enable_gradient_checkpointing()

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))
    
    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        distill: bool = False,
        compute_viewspace_depth: bool = True,
        **kwargs,
    ) -> dict:
        device = context["image"].device
        B, T, _, H, W = context["image"].shape
        spatial_shape = torch.tensor(context["image"].shape[-2:])

        ## Encode the input video to latent space
        input_video = context["image"].permute(0, 2, 1, 3, 4)

        gs_embeds, camera_embeds, global_embeds, interms = self.backbone(
            input_video, context["intrinsics"]
        )

        # predict camera pose
        pred_extrins = self.camera_extrinsic_head(camera_embeds)
        # pred_extrins = rearrange(pred_extrins, "b t (k c) -> b (t k) c", k=1)
        pred_extrins[..., 3] = pred_extrins[..., 3] + 1.0
        if self.cfg.camera_type == "dq":
            # normalize the dq array
            r_norm = pred_extrins[..., :4].norm(dim=-1, keepdim=True)
            pred_extrins = pred_extrins / r_norm
            pred_extrinsics_4x4 = camera_matrix_from_dq_array(pred_extrins)
        else:
            q, t = pred_extrins.split((4, 3), dim=-1)
            q = torch.nn.functional.normalize(q, dim=-1)
            pred_extrins = torch.cat([q, t], dim=-1)
            pred_extrinsics_4x4 = camera_matrix_from_qt_array(pred_extrins)

        pred_extrinsics_4x4 = torch.cat(
            [context["extrinsics"][:, :1], pred_extrinsics_4x4], dim=1
        )

        # maybe predict camera intrinsics
        if global_embeds is not None:
            pred_intrins = self.camera_intrinsic_head(global_embeds)
            pred_intrinsics_3x3 = simple_intrin_matrix_from_fov(pred_intrins)
            pred_intrinsics_3x3 = pred_intrinsics_3x3.unsqueeze(1).repeat(1, T, 1, 1)
        else:
            pred_intrins = None
            pred_intrinsics_3x3 = None

        if self.cfg.gs_center_head_type == "dpt":
            gs_center_head_out = self.head1(
                [rearrange(interm, "b t n c -> (b t) n c") for interm in interms], 
                spatial_shape
            )
            gs_centers = rearrange(gs_center_head_out['pts3d'], "(b t) h w c -> b t h w c", b=B)
            if self.predict_confidence:
                conf = rearrange(gs_center_head_out['conf'], "(b t) h w -> b t h w", b=B)
            else:
                conf = None
        else:
            raise ValueError(f"Unsupported gaussian center head type {self.cfg.gs_center_head_type}.")

        if compute_viewspace_depth:
            # project canonical space points back to their corresponding view space
            viewspace_pts = torch.einsum(
                "bvij,bvhwjk->bvhwik", 
                context["extrinsics"][:, :, :3, :3].inverse(),
                (gs_centers - context["extrinsics"][:, :, None, None, :3, 3]).unsqueeze(-1)
            ).squeeze(-1).contiguous()
            viewspace_depth = viewspace_pts[..., -1]    # z-axis
        else:
            viewspace_depth = None
        
        if distill:
            return dict(
                pred_extrins=pred_extrins,
                pred_intrins=pred_intrins,
                gaussian_camera_extrins=pred_extrinsics_4x4,
                gaussian_camera_intrins=pred_intrinsics_3x3,
                gaussian_centers=gs_centers,
                confidence=conf,
                context_view_depths=viewspace_depth,
            )
        
        if self.cfg.gs_param_head_type == "dpt_gs":
            gs_params = self.gaussian_param_head(
                [rearrange(interm, "b t n c -> (b t) n c") for interm in interms], 
                None, 
                rearrange(input_video, "b c t h w -> (b t) c h w"), 
                spatial_shape
            )
            gs_params = rearrange(gs_params, "(b t) c h w -> b t h w c", b=B)
        else:
            raise ValueError(f"Unsupported gaussian param head type {self.cfg.gs_param_head_type}.")

        raw_gaussians = torch.cat([gs_centers, gs_params], dim=-1)


        gaussians = self.gaussian_adapter.forward(
            raw_gaussians, 
            pdf_to_opacity_func=None if self.cfg.predict_opacity else lambda o: self.map_pdf_to_opacity(o, global_step)
        )
        if visualization_dump is not None:
            visualization_dump["depth"] = gaussians.means[..., -1:]

        # import pdb; pdb.set_trace()
        
        return dict(
            gaussians=gaussians,
            pred_extrins=pred_extrins,
            pred_intrins=pred_intrins,
            raw_gaussians=raw_gaussians,
            gaussian_camera_extrins=pred_extrinsics_4x4,
            gaussian_camera_intrins=pred_intrinsics_3x3,
            gaussian_centers=gs_centers,
            confidence=conf,
            context_view_depths=viewspace_depth,
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
