from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
import torch
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss, l1_loss, l2_loss

from ..misc.dq import DualQuaternion
from ..misc.cam_utils import camera_dq_array_from_Rt, camera_q_from_R
from ..geometry.projection import get_fov


@dataclass
class LossCameraCfg:
    weight: float
    use_dq_loss: bool = True
    camera_type: Literal["dq", "qt"] = "dq"


@dataclass
class LossCameraCfgWrapper:
    camera: LossCameraCfg


def camera_dq_loss(prediction, target):
    if isinstance(prediction, torch.Tensor):
        prediction = DualQuaternion.from_dq_array(prediction)
    if isinstance(target, torch.Tensor):
        target = DualQuaternion.from_dq_array(target)

    identity_dq = DualQuaternion.identity(
        dq_size=prediction.q_r.lshape
    ).dq_array.to(device=prediction.q_r.device, dtype=prediction.q_r.dtype)
    
    loss = l1_loss(
        (prediction * target.conjugate).dq_array, identity_dq
    ) + l1_loss(
        (target * prediction.conjugate).dq_array, identity_dq
    )
    return loss

class LossCamera(Loss[LossCameraCfg, LossCameraCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        pred_camera_array = prediction.extrinsics # (batch view 8)
        pred_intrins = prediction.intrinsics    # (batch 2)

        context_extrins_4x4 = batch["context"]["extrinsics"][:, 1:]
        if self.cfg.camera_type == "dq":
            assert pred_camera_array.shape[-1] == 8
            context_camera_array = camera_dq_array_from_Rt(
                context_extrins_4x4[..., :3, :3], context_extrins_4x4[..., :3, 3]
            )
        else:
            assert pred_camera_array.shape[-1] == 7
            context_camera_array = torch.cat(
                [
                    camera_q_from_R(context_extrins_4x4[..., :3, :3])[..., [1, 2, 3, 0]],
                    context_extrins_4x4[..., :3, 3]
                ], dim=-1
            )   # (b, v, xyzr_xyz)
        if self.cfg.use_dq_loss and self.cfg.camera_type == "dq":
            dq_loss = camera_dq_loss(pred_camera_array, context_camera_array)
            loss_camera = dq_loss + l1_loss(pred_camera_array, context_camera_array)
        else:
            loss_camera = l1_loss(pred_camera_array, context_camera_array)
        if pred_intrins is not None:
            context_intrins = get_fov(batch["context"]["intrinsics"].mean(dim=1))
            loss_camera = loss_camera + l2_loss(pred_intrins, context_intrins)
        return self.cfg.weight * loss_camera

