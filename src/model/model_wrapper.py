from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable, Any

import json
import numpy as np
import moviepy.editor as mpy
import torch
import wandb
import plotly
import imageio
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim
from torchvision.io import write_video

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim, camera_eval_metrics
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_conf_point import Regr3D
from ..loss.loss_ssim import ssim
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose, get_pnp_pose, camera_matrix_from_dq_array, simple_intrin_matrix_from_fov
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger, LocalTensorboardLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.drawing.cameras import create_plotly_cameras_visualization
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode, DecoderOutput
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .ply_export import export_ply

@dataclass
class WrappedOutput(DecoderOutput):
    extrinsics: Float[Tensor, "batch view 8"] | None = None  # dual-quaternions
    intrinsics: Float[Tensor, "batch 2"] | None = None    # fovx & fovy


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    save_gs: bool = False


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_only_steps: int
    distill_max_steps: int
    distill_weight: float = 0.1
    gradient_checkpointing: bool = False
    sh_warmup_every_n_steps: Optional[int] = -1
    n_camera_opt_views: Optional[int] = 0
    lr_cosine_annealing: bool = True
    new_param_keywords: Optional[list[str]] = None


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        distiller: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        self.distiller = distiller
        self.distiller_loss = None
        if self.distiller is not None:
            convert_to_buffer(self.distiller, persistent=False)
            self.distiller_loss = Regr3D()

        # This is used for testing.
        self.benchmarker = Benchmarker()

    def _sample_anchor_frames(
        self, video_frames: Float[Tensor, "B V C H W"], temporal_compression: int = 4, n_frames: Optional[int] = None,
    ) -> Float[Tensor, "B v C H W"]:
        ## If given odd number of frames, sample anchor frames after the first frame
        ## with the first frame treat as reference frame
        B, video_length, C, H, W = video_frames.shape
        is_odd = video_length % 2 != 0

        if not is_odd: 
            n_segments = video_length // temporal_compression
        else:
            n_segments = (video_length - 1) // temporal_compression + 1

        n_frames = n_frames or n_segments
        assert n_frames == 2
        idx = torch.from_numpy( # frame idx in the picked segment
            np.random.choice(temporal_compression, (B, n_frames,),)
        ).long().to(video_frames.device)
        segment_idx = torch.from_numpy(np.stack(
            [np.random.choice(n_segments-1, (n_frames-1,), replace=False) for _ in range(B)], axis=0
        )).long().to(video_frames.device)
        segment_idx = torch.cat([segment_idx, segment_idx+1], dim=-1)
        idx = idx + segment_idx * temporal_compression
        if is_odd:
            idx = (idx + 1 - temporal_compression).clamp_min(0).long()
        # idx = torch.from_numpy(idx).long().to(video_frames.device)
        anchor_frames = torch.gather(
            video_frames, dim=1, index=idx[..., None, None, None].repeat(1, 1, C, H, W)
        )
        return anchor_frames, idx, segment_idx

    def training_step(self, batch, batch_idx):
        distill_only = self.distiller is not None and self.global_step < self.train_cfg.distill_only_steps
        # combine batch from different dataloaders
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        total_loss = 0.0
        # Run the model.
        model_outs = self.encoder(batch["context"], self.global_step, distill=distill_only)

        context_gt = batch["context"]["image"]
        # Render
        if not distill_only:
            render_pkg = self.decoder.forward(
                model_outs["gaussians"],
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )
            output = WrappedOutput(
                color=render_pkg.color,
                depth=render_pkg.depth,
                extrinsics=model_outs["pred_extrins"],
                intrinsics=model_outs["pred_intrins"]
            )
            target_gt = batch["target"]["image"]

            # Compute metrics.
            psnr_probabilistic = compute_psnr(
                rearrange(target_gt, "b v c h w -> (b v) c h w"),
                rearrange(output.color, "b v c h w -> (b v) c h w"),
            )
            self.log("train/psnr_probabilistic", psnr_probabilistic.mean())
        else:
            output = WrappedOutput(
                extrinsics=model_outs["pred_extrins"],
                intrinsics=model_outs["pred_intrins"]
            )

        # Compute and log loss.
        for loss_fn in self.losses:
            if not distill_only or loss_fn.name == "camera":
                loss = loss_fn.forward(output, batch, model_outs, self.global_step)
                self.log(f"loss/{loss_fn.name}", loss)
                total_loss = total_loss + loss

        # distillation
        if self.distiller is not None and self.global_step <= self.train_cfg.distill_max_steps:
            # randomly sample two segments and sample one frame from each as distiller input
            anchor_frames, frame_idx, segment_idx = self._sample_anchor_frames(
                batch["context"]["image"], n_frames=2, temporal_compression=1
            )
            # pairs = torch.cat(
            #     [batch["context"]["image"][:, :1], anchor_frames], dim=1
            # )
            distiller_input = {"image": anchor_frames}
            with torch.no_grad():
                pseudo_gt1, pseudo_gt2 = self.distiller(distiller_input, False)
            # currently the predicted pseudo gt is in the canonical space of the first sampled frame
            # we need to transform them to the canonical space of the first video frame
            first_anchor_extrins = torch.gather(
                batch["context"]["extrinsics"], dim=1,
                index=frame_idx[:, :1, None, None].repeat(1, 1, 4, 4)
            ).squeeze(1)
            pseudo_gt_pts1 = torch.einsum(
                "bij,bhwjk->bhwik", first_anchor_extrins[:, :3, :3], pseudo_gt1['pts3d'].unsqueeze(-1)
            ).squeeze(-1) + first_anchor_extrins[:, None, None, :3, -1]
            pseudo_gt_pts2 = torch.einsum(
                "bij,bhwjk->bhwik", first_anchor_extrins[:, :3, :3], pseudo_gt2['pts3d'].unsqueeze(-1)
            ).squeeze(-1) + first_anchor_extrins[:, None, None, :3, -1]

            # obtain corresponding predicted Gaussian xyz
            gaussians_xyz = model_outs["gaussian_centers"]
            gaussians_conf = model_outs["confidence"]
            pred_pts = torch.gather(
                gaussians_xyz, 
                dim=1,
                index=segment_idx[..., None, None, None].repeat(1, 1, *gaussians_xyz.shape[-3:])
            )
            pred_pts1, pred_pts2 = torch.unbind(pred_pts, dim=1)
            if gaussians_conf is not None:
                pred_conf = torch.gather(
                    gaussians_conf, 
                    dim=1,
                    index=segment_idx[..., None, None].repeat(1, 1, *gaussians_conf.shape[-2:])
                )
                pred_conf1, pred_conf2 = torch.unbind(pred_conf, dim=1)
            else:
                pred_conf1 = pred_conf2 = None
            
            # calculate the distillation loss
            distillation_loss = self.distiller_loss.forward(
                pseudo_gt_pts1, pseudo_gt_pts2, 
                pred_pts1, pred_pts2,
                pseudo_gt1['conf'], pseudo_gt2['conf'], 
                pred_conf1, pred_conf2,
                normalize_pts=context_gt.shape[1] > 2
            ) * self.train_cfg.distill_weight
            self.log("loss/distillation_loss", distillation_loss)
            total_loss = total_loss + distillation_loss

        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            model_outs = self.encoder(
                batch["context"],
                self.global_step,
            )
            gaussians = model_outs["gaussians"]
            
        if self.test_cfg.align_pose:
            output = self.test_step_align(batch, gaussians)
        else:
            with self.benchmarker.time("decoder", num_calls=v):
                output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                )

        # compute scores
        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0]
            overlap_tag = get_overlap_tag(overlap)

            # photometrics
            rgb_pred = output.color[0]
            rgb_gt = batch["target"]["image"][0]
            all_metrics = {
                f"lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean(),
                f"ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean(),
                f"psnr_ours": compute_psnr(rgb_gt, rgb_pred).mean(),
            }
            methods = ['ours']

            # camera pose metrics
            pred_extrinsics = model_outs["gaussian_camera_extrins"][0]
            # pred_intrinsics = model_outs["gaussian_camera_intrins"][0]
            gt_extrinsics = batch["context"]["extrinsics"][0]

            # Compute camera pose evaluation metrics
            # ATE, RPE-trans, RPE-rot
            try:
                ate, rpe_trans, rpe_rot = camera_eval_metrics(pred_extrinsics, gt_extrinsics)
            except:
                ate = rpe_trans = rpe_rot = 0.0

            all_metrics.update({
                "ate_ours": ate, "rpe-trans_ours": rpe_trans, "rpe-rot_ours": rpe_rot
            })

            self.log_dict(all_metrics)
            self.print_preview_metrics(all_metrics, methods, overlap_tag=None)

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name

        # Always save transforms
        frames = []
        for i, (index, color) in enumerate(zip(batch["context"]["index"][0], batch["context"]["image"][0])):
            save_image(color*0.5+0.5, path / scene / f"context/{index:0>6}.png")
            frame = {
                "file_path": f"context/{index:0>6}.png",
                "transform_matrix": model_outs["gaussian_camera_extrins"][0, i].cpu().numpy().tolist()
            }
            frames.append(frame)
        with open(path / scene / "transforms.json", "w") as f:
            json.dump(frames, f, indent=4)

        # Save images. both rgb and depth
        if self.test_cfg.save_image:
            depth_pred = vis_depth_map(output.depth[0])
            for i, (index, color, depth) in enumerate(zip(batch["target"]["index"][0], output.color[0], depth_pred)):
                img_to_save = hcat(color, depth)
                save_image(img_to_save, path / scene / f"color/{index:0>6}.png")

        if self.test_cfg.save_video:
            video_save_path = str(path / scene / "interpolation.mp4")
            self.render_video_interpolation(
                batch, 
                override_save_path=video_save_path, 
                save_depth=False, 
                fps=30,
                num_frames=60,
            )

        if self.test_cfg.save_compare:
            # Construct comparison image.
            context_img = inverse_normalize(batch["context"]["image"][0])
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
            )
            save_image(comparison, path / f"{scene}.png")

        ## Export Gaussians
        if self.test_cfg.save_gs:
            export_ply(
                0,
                means=gaussians.means[0].flatten(0, 2),
                scales=gaussians.scales[0].flatten(0, 2),
                rotations=gaussians.rotations[0].flatten(0, 2),
                harmonics=gaussians.harmonics[0].flatten(0, 2),
                opacities=gaussians.opacities[0].flatten(),
                path=path / scene / "gaussians.ply",
                save_sh_dc_only=True    # only save the first sh channel to avoid too large size
            )
            
    def test_step_align(self, batch, gaussians):
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["target"]["image"].shape
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

            opt_params = []
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": self.test_cfg.rot_opt_lr,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": self.test_cfg.trans_opt_lr,
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)

            extrinsics = batch["target"]["extrinsics"].clone()
            with self.benchmarker.time("optimize"):
                for i in range(self.test_cfg.pose_align_steps):
                    pose_optimizer.zero_grad()

                    output = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        if loss_fn.name != "camera":
                            loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                            total_loss = total_loss + loss

                    total_loss.backward()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                    )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)

        # Render Gaussians.
        output = self.decoder.forward(
            gaussians,
            extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )

        return output

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.summarize()
        
    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)
        distill_only = self.distiller is not None and self.global_step < self.train_cfg.distill_only_steps
        label = f"dataset{dataloader_idx}"
        
        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"{dataloader_idx=}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        visualization_dump = {}
        model_outs = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
            distill=distill_only,
            compute_viewspace_depth=True,
        )

        # Construct comparison image.
        context_img = inverse_normalize(batch["context"]["image"][0])
        context_img_depth = vis_depth_map(model_outs["context_view_depths"][0])

        context = []
        context_pseudo_depths = []
        for i in range(context_img.shape[0]):
            context.append(context_img[i])
            context_pseudo_depths.append(context_img_depth[i])

        if distill_only:
            with torch.no_grad():
                pseudo_gt1, pseudo_gt2 = self.distiller(batch["context"], False)
            pseudo_gt_pts1, pseudo_gt_pts2 = pseudo_gt1['pts3d'], pseudo_gt2['pts3d']
            pseudo_gt_depth1 = pseudo_gt_pts1[..., -1]
            pseudo_gt_depth2 = torch.einsum(
                "bij,bhwjk->bhwik", 
                batch["context"]["extrinsics"][:, 1, :3, :3].inverse(), 
                (pseudo_gt_pts2 - batch["context"]["extrinsics"][:, 1, None, None, :3, -1]).unsqueeze(-1)
            ).squeeze(-1)[..., -1]
            pseudo_gt_conf1, pseudo_gt_conf2 = pseudo_gt1['conf'], pseudo_gt2['conf']
            pred_ctxt_view_depth = model_outs["context_view_depths"][0]
            pred_conf = model_outs["confidence"][0]
            # viz
            pseudo_gt_depth = vis_depth_map(torch.cat([pseudo_gt_depth1, pseudo_gt_depth2]))
            pseudo_gt_conf = confidence_map(torch.cat([pseudo_gt_conf1, pseudo_gt_conf2]))
            pred_depth = vis_depth_map(pred_ctxt_view_depth)
            comparison = hcat(
                add_label(vcat(*context), "Context Image"),
                add_label(vcat(*pseudo_gt_depth), "DUSt3R Depth"),
                add_label(vcat(*pseudo_gt_conf), "DUSt3R Confidence"),
                add_label(vcat(*pred_depth), "Pred Depth"),
            )
            if model_outs["confidence"] is not None:
                pred_conf = model_outs["confidence"][0]
                pred_conf = confidence_map(pred_conf)
                comparison = hcat(comparison, add_label(vcat(*pred_conf), "Pred Confidence"))
        else:
            gaussians = model_outs["gaussians"]
            render_pkg = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
            )
            output = WrappedOutput(
                color=render_pkg.color,
                depth=render_pkg.depth,
                extrinsics=model_outs["pred_extrins"],
                intrinsics=model_outs["pred_intrins"]
            )
            rgb_pred = output.color[0]
            depth_pred = vis_depth_map(output.depth[0])

            # Compute validation metrics.
            rgb_gt = batch["target"]["image"][0]
            psnr = compute_psnr(rgb_gt, rgb_pred).mean()
            self.log(f"val/{label}/psnr", psnr)
            lpips = compute_lpips(rgb_gt, rgb_pred).mean()
            self.log(f"val/{label}/lpips", lpips)
            ssim = compute_ssim(rgb_gt, rgb_pred).mean()
            self.log(f"val/{label}/ssim", ssim)

            comparison = hcat(
                add_label(vcat(*context), "Context Image"),
                add_label(vcat(*context_pseudo_depths), "Context View Depth"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
                add_label(vcat(*depth_pred), "Depth (Prediction)"),
            )

            # Render projections and construct projection image.
            # These are disabled for now, since RE10k scenes are effectively unbounded.
            projections = hcat(
                    *render_projections(
                        gaussians,
                        256,
                        extra_label="",
                    )[0]
                )
            self.logger.log_image(
                f"{label}/projection",
                [prep_image(add_border(projections))],
                step=self.global_step,
                dataformats="HWC",
            )

            if self.encoder_visualizer is not None:
                for k, image in self.encoder_visualizer.visualize(
                    batch["context"], self.global_step
                ).items():
                    self.logger.log_image(k, [prep_image(image)], step=self.global_step, dataformats="HWC")

            # Run video validation step.
            self.render_video_interpolation(batch)
            # self.render_video_wobble(batch)
            if self.train_cfg.extended_visualization:
                self.render_video_interpolation_exaggerated(batch)

        self.logger.log_image(
            f"{label}/comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            dataformats="HWC",
            caption=batch["scene"],
        )

        gt_extrinsics = batch["context"]["extrinsics"][0]
        gt_intrinsics = batch["context"]["intrinsics"][0]
        nears = batch["context"]["near"][0]
        fars = batch["context"]["far"][0]
        pred_extrinsics = model_outs["gaussian_camera_extrins"][0]
        try:
            pred_intrinsics = model_outs["gaussian_camera_intrins"][0]
        except:
            pred_intrinsics = gt_intrinsics

        ## Draw cameras
        traj_skip = 1
        camera_traj_vis = create_plotly_cameras_visualization(
            cameras_gt=dict(extrins=gt_extrinsics[::traj_skip].clone(), intrins=gt_intrinsics[::traj_skip].clone()),
            cameras_pred=dict(extrins=pred_extrinsics[::traj_skip].clone(), intrins=pred_intrinsics[::traj_skip].clone()),
        )
        self.logger.log_image(
            f"{label}/camera_traj",
            [camera_traj_vis],
            step=self.global_step,
            dataformats="HWC",
        )

        torch.cuda.empty_cache()


    # @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        # if v != 2:
        #     return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, -1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)
    
    # @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample, name_suffix: str = None, **kwargs) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                batch["context"]["extrinsics"][0, -1],
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, -1],
                t,
            )
            return extrinsics[None], intrinsics[None]
        
        name = "interpolation"
        if name_suffix is not None:
            name = name + f"_{name_suffix}"

        return self.render_video_generic(batch, trajectory_fn, name, **kwargs)

    # @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample, name_suffix: str = None, **kwargs) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        # if v != 2:
        #     return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, -1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                batch["context"]["extrinsics"][0, -1],
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, -1],
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        name = "interpolation_exagerrated"
        if name_suffix is not None:
            name = name + f"_{name_suffix}"
            
        return self.render_video_generic(
            batch,
            trajectory_fn,
            name,
            num_frames=300,
            smooth=False,
            loop_reverse=False,
            **kwargs,
        )

    # @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
        fps: int = 30,
        save_depth: bool = True,
        override_save_path: Optional[str] = None,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians = self.encoder(batch["context"], self.global_step)["gaussians"]

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth",
        )
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ] if save_depth else [
            rgb for rgb in output.color[0]
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=fps, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, (LocalLogger, LocalTensorboardLogger))
            for key, value in visualizations.items():
                dir = Path(self.logger.save_dir) / key
                dir.mkdir(exist_ok=True, parents=True)
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=fps)
                clip.write_videofile(
                    override_save_path or str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim", "ate", "rpe-trans", "rpe-rot"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # if "backbone" in name or "head" in name:
            #     new_params.append(param)
            #     new_param_names.append(name)
            # else:
            #     pretrained_params.append(param)
            #     pretrained_param_names.append(name)
            is_new_param = False
            if self.train_cfg.new_param_keywords:
                for keyword in self.train_cfg.new_param_keywords:
                    if keyword in name:
                        is_new_param = True
                        break

            if is_new_param:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        if len(new_param_names) > 0:
            param_dicts = [
                {
                    "params": new_params,
                    "lr": self.optimizer_cfg.lr,
                },
                {
                    "params": pretrained_params,
                    "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
                }
            ]
        else:
            param_dicts = [
                {
                    "params": pretrained_params,
                    "lr": self.optimizer_cfg.lr,
                }
            ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        if self.train_cfg.lr_cosine_annealing:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])
        else:
            lr_scheduler = warm_up

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
