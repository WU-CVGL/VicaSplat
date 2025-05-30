from dataclasses import dataclass
from typing import Literal, Optional

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_gsplat
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool
    use_gsplat: bool = True


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
        use_sh: bool = True,
        active_sh_degree: Optional[int] = None,
        return_dict: bool = True,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        if gaussians.means.ndim > 3:    # need flatten
            gaussians = Gaussians(
                gaussians.means.flatten(1, 3),
                gaussians.covariances.flatten(1, 3),
                gaussians.harmonics.flatten(1, 3),
                gaussians.opacities.flatten(1)
            )
        if self.cfg.use_gsplat:
            color, depth = render_gsplat(
                extrinsics,
                intrinsics,
                near,
                far,
                image_shape,
                repeat(self.background_color, "c -> b v c", b=b, v=v),
                gaussians.means,
                gaussians.covariances,
                gaussians.harmonics,
                gaussians.opacities,
                scale_invariant=self.make_scale_invariant,
                use_sh=use_sh,
                sh_degree=active_sh_degree,
                frame_by_frame=image_shape[0]>=512,
            )
        else:
            color, depth = render_cuda(
                rearrange(extrinsics, "b v i j -> (b v) i j"),
                rearrange(intrinsics, "b v i j -> (b v) i j"),
                rearrange(near, "b v -> (b v)"),
                rearrange(far, "b v -> (b v)"),
                image_shape,
                repeat(self.background_color, "c -> (b v) c", b=b, v=v),
                repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
                repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
                repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
                repeat(gaussians.opacities, "b g -> (b v) g", v=v),
                scale_invariant=self.make_scale_invariant,
                cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i") if cam_rot_delta is not None else None,
                cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i") if cam_trans_delta is not None else None,
                use_sh=use_sh,
                sh_degree=active_sh_degree,
            )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        depth = rearrange(depth, "(b v) h w -> b v h w", b=b, v=v)
        if not return_dict:
            return color, depth
        return DecoderOutput(color, depth)
