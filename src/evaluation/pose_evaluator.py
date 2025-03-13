import json
import os
import sys
from typing import Any

import math
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..misc.cam_utils import camera_normalization, pose_auc, update_pose, get_pnp_pose

import csv
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningModule
from tabulate import tabulate

from ..loss.loss_ssim import ssim
from ..misc.image_io import load_image, save_image
from ..misc.utils import inverse_normalize, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from .evaluation_cfg import EvaluationCfg
from .metrics import compute_lpips, compute_psnr, compute_ssim, compute_pose_error, camera_eval_metrics


class PoseEvaluator(LightningModule):
    cfg: EvaluationCfg

    def __init__(self, cfg: EvaluationCfg, encoder, decoder, losses) -> None:
        super().__init__()
        self.cfg = cfg

        # our model
        self.encoder = encoder.to(self.device)
        self.decoder = decoder
        self.losses = nn.ModuleList(losses)

        self.data_shim = get_data_shim(self.encoder)

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        # set to eval
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # get overlap.
        overlap = batch["context"]["overlap"][0, 0]
        overlap_tag = get_overlap_tag(overlap)
        if overlap_tag == "ignore":
            return

        # runing encoder to obtain the 3DGS
        # input_images_view2 = batch["context"]["image"][:, 1:2].clone()
        # input_images_view2 = input_images_view2 * 0.5 + 0.5
        # visualization_dump = {}
        # gaussians = self.encoder(
        #     batch["context"],
        #     self.global_step,
        #     visualization_dump=visualization_dump,
        # )
        input_images_otherview = batch["context"]["image"][:, 1:].clone()
        input_images_otherview = input_images_otherview * 0.5 + 0.5
        with torch.no_grad():
            model_outs = self.encoder(batch["context"], self.global_step)
        gaussians = model_outs["gaussians"]
        pred_extrinsics = model_outs["gaussian_camera_extrins"]
        pose_opt = pred_extrinsics.clone()[:, 1:]

        # # optimize the pose using PnPRansac
        # pose_opt = get_pnp_pose(visualization_dump['means'][0, 1].squeeze(),
        #                         visualization_dump['opacities'][0, 1].squeeze(),
        #                         batch["context"]["intrinsics"][0, 1], h, w)
        # pose_opt = pose_opt.to(self.device)
        # # pose_opt = batch["context"]["extrinsics"][0, 0].clone()  # initial pose as the first view: I

        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v-1, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v-1, 3], requires_grad=True, device=self.device))

            opt_params = []
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": 0.005,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": 0.005,
                }
            )

            pose_optimizer = torch.optim.Adam(opt_params)

            number_steps = 0 # 200
            extrinsics = pose_opt
            for i in range(number_steps):
                pose_optimizer.zero_grad()

                output = self.decoder.forward(
                    gaussians,
                    extrinsics,
                    batch["context"]["intrinsics"][:, 1:],
                    batch["context"]["near"][:, 1:],
                    batch["context"]["far"][:, 1:],
                    (h, w),
                    cam_rot_delta=cam_rot_delta,
                    cam_trans_delta=cam_trans_delta,
                )

                # Compute and log loss.
                batch["target"]["image"] = input_images_otherview
                total_loss = 0
                for loss_fn in self.losses:
                    if loss_fn.name != "camera":
                        loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                        total_loss = total_loss + loss

                # add ssim structure loss
                ssim_, _, _, structure = ssim(rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
                                      rearrange(output.color, "b v c h w -> (b v) c h w"),
                                      size_average=True, data_range=1.0, retrun_seprate=True, win_size=11)
                ssim_loss = (1 - structure) * 1.0
                total_loss = total_loss + ssim_loss

                # backpropagate
                # print(f"Step {i} - Loss: {total_loss.item()}")
                total_loss.backward()
                with torch.no_grad():
                    pose_optimizer.step()
                    new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                )
                    cam_rot_delta.data.fill_(0)
                    cam_trans_delta.data.fill_(0)

                    extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b)

            # eval pose
            gt_pose = batch["context"]["extrinsics"][0]
            eval_pose = torch.cat([batch["context"]["extrinsics"][:, :1], extrinsics], dim=1)[0]
            # error_t, error_t_scale, error_R = compute_pose_error(gt_pose, eval_pose)    # noposplat
            # error_pose = torch.max(error_t, error_R)  # find the max error
            ate, rpe_trans, rpe_rot = camera_eval_metrics(eval_pose, gt_pose)

            pred_pose = pred_extrinsics[0]
            ate_, rpe_trans_, rpe_rot_ = camera_eval_metrics(pred_pose, gt_pose)

            all_metrics = {
                "ate_after_opt": ate,
                "rpe_trans_after_opt": rpe_trans,
                "rpe_rot_after_opt": rpe_rot,
                "ate_before_opt": ate_,
                "rpe_trans_before_opt": rpe_trans_,
                "rpe_rot_before_opt": rpe_rot_
            }

            # self.log_dict(all_metrics)
            self.print_preview_metrics(all_metrics, None)

            return 0

    # def calculate_auc(self, tot_e_pose, method_name, overlap_tag):
    #     thresholds = [5, 10, 20]
    #     auc = pose_auc(tot_e_pose, thresholds)
    #     print(f"Pose AUC {method_name} {overlap_tag}: ")
    #     print(auc)
    #     return auc

    # def on_test_end(self) -> None:
    #     # eval pose
    #     for method in self.cfg.methods:
    #         tot_e_pose = np.array(self.all_mertrics[f"e_pose_{method.key}"])
    #         tot_e_pose = np.array(tot_e_pose)
    #         thresholds = [5, 10, 20]
    #         auc = pose_auc(tot_e_pose, thresholds)
    #         print(f"Pose AUC {method.key}: ")
    #         print(auc)

    #         for overlap_tag in self.all_mertrics_sub.keys():
    #             tot_e_pose = np.array(self.all_mertrics_sub[overlap_tag][f"e_pose_{method.key}"])
    #             tot_e_pose = np.array(tot_e_pose)
    #             thresholds = [5, 10, 20]
    #             auc = pose_auc(tot_e_pose, thresholds)
    #             print(f"Pose AUC {method.key} {overlap_tag}: ")
    #             print(auc)

    #     # save all metrics
    #     np.save("all_metrics.npy", self.all_mertrics)
    #     np.save("all_metrics_sub.npy", self.all_mertrics_sub)

    def print_preview_metrics(self, metrics: dict[str, float], overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1

            self.all_mertrics = {k: [v.cpu().item() if isinstance(v, torch.Tensor) else v] for k, v in metrics.items()}
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

            for k, v in metrics.items():
                self.all_mertrics[k].append(v.cpu().item() if isinstance(v, torch.Tensor) else v)

        metric_list = [
            "ate", "rpe_trans", "rpe_rot", 
        ]

        def print_metrics(runing_metric):
            table = []
            for method in ["before_opt", "after_opt"]:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics)