import os
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from einops import rearrange, repeat
from jaxtyping import Float
from tqdm import tqdm

eval_index_2v_file_path = "./assets/evaluation_index_scannet.json"
data_root = Path("../../dataset/scannet/scannet")
stage = "test"

n_context_views = 8
n_target_views = 9

# 1. read scannet dataset
# 2. get the scene key and scene total views
# 3. expand the eval index


def get_chunk_paths() -> list[str]:
    root = data_root / stage
    root_chunks = sorted(
        [path for path in root.iterdir()]
    )
    return root_chunks

if __name__ == "__main__":
    chunk_paths = get_chunk_paths()

    # Load 2-view index file
    with open(eval_index_2v_file_path, "r") as f:
        index = json.load(f)

    new_index_dict = {}
    for chunk_path in tqdm(chunk_paths):
        scene = str(chunk_path).split('/')[-1]
        extrinsics = torch.from_numpy(np.load(os.path.join(chunk_path, 'extrinsics.npy'))).float()
        n_total_views = extrinsics.shape[0]
        
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
    
    with open(f"assets/evaluation_index_scannet_{n_context_views}v.json", "w") as f:
        json.dump(new_index_dict, f, indent=4)