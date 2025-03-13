import os
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from einops import rearrange, repeat
from jaxtyping import Float
from tqdm import tqdm

eval_index_2v_file_path = "./evaluation_index_re10k.json"
data_root = Path("/run/determined/workdir/SSD/re10k/re10k")
stage = "test"

n_context_views = 4
n_target_views = 9
n_target_views_every_two_context_views = 2


def get_chunk_paths() -> list[dict]:
    root = data_root / stage
    root_chunks = sorted(
        [path for path in root.iterdir() if path.suffix == ".torch"]
    )
    return root_chunks


def convert_poses(
    self,
    poses: Float[Tensor, "batch 18"],
) -> tuple[
    Float[Tensor, "batch 4 4"],  # extrinsics
    Float[Tensor, "batch 3 3"],  # intrinsics
]:
    b, _ = poses.shape

    # Convert the intrinsics to a 3x3 normalized K matrix.
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intrinsics


if __name__ == "__main__":
    chunk_paths = get_chunk_paths()

    # Load 2-view index file
    with open(eval_index_2v_file_path, "r") as f:
        index = json.load(f)

    new_index_dict = {}
    for chunk_path in tqdm(chunk_paths):
        chunk = torch.load(chunk_path)

        for example in chunk:
            scene = example["key"]

            n_total_views = example["cameras"].shape[0]

            scene_index = index.get(scene)

            if scene_index is None:
                print(f"skip missing scene {scene}")
                continue

            ctxt_idx_1, ctxt_idx_2 = scene_index["context"]
            ctxt_interval = ctxt_idx_2 - ctxt_idx_1

            if 1 + ctxt_interval * (n_context_views - 1) > n_total_views:
                ctxt_interval = (n_total_views - 1) // (n_context_views - 1)

            if ctxt_idx_1 + ctxt_interval * (n_context_views - 1) > n_total_views - 1:
                ctxt_idx_start_max = n_total_views - 1 - ctxt_interval * (n_context_views - 1)
                ctxt_idx_start = np.random.randint(0, ctxt_idx_start_max + 1)
            else:
                ctxt_idx_start = ctxt_idx_1
            
            ctxt_indices = np.arange(n_context_views) * ctxt_interval + ctxt_idx_start

            ctxt_idx_start, ctxt_idx_end = ctxt_indices[0], ctxt_indices[-1]
            assert ctxt_idx_end < n_total_views

            tgt_indices = np.random.choice(ctxt_idx_end - ctxt_idx_start + 1, (n_target_views,), replace=False) + ctxt_idx_start
            tgt_indices = np.sort(tgt_indices)

            new_index_dict[scene] = {
                "context": ctxt_indices.tolist(),
                "target": tgt_indices.tolist(),
                "overlap": 0.5,
            }

    with open(f"assets/evaluation_index_re10k_{n_context_views}v.json", "w") as f:
        json.dump(new_index_dict, f, indent=4)




