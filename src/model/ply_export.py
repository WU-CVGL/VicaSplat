from pathlib import Path
from typing import Optional

import numpy as np
import torch
from einops import einsum, rearrange
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes

def inverse_sigmoid(x):
    return torch.log(x/(1-x))



def export_ply(
    extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
    save_sh_dc_only: bool = False,
):
    # prune by opacity
    mask = opacities >= 0.005
    opacities = opacities[mask]
    opacities, indices = torch.sort(opacities, descending=True)
    means = means[mask][indices]
    rotations = rotations[mask][indices]
    scales = scales[mask][indices]
    harmonics = harmonics[mask][indices]

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since our axes are swizzled for the spherical harmonics, we only export the DC
    # band.
    # harmonics_view_invariant = harmonics[..., 0]
    # print(harmonics_view_invariant.shape)
    f_dc = harmonics[..., 0]
    f_rest = harmonics[..., 1:].flatten(start_dim=1)

    dtype_full = [
        (attribute, "f4") for attribute in construct_list_of_attributes(
            0 if save_sh_dc_only else f_rest.shape[1])
    ]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    if save_sh_dc_only:
        attributes = (
            means.detach().cpu().numpy(),
            torch.zeros_like(means).detach().cpu().numpy(),
            f_dc.detach().cpu().contiguous().numpy(),
            inverse_sigmoid(opacities)[..., None].detach().cpu().numpy(),
            scales.log().detach().cpu().numpy(),
            rotations,
        )
    else:
        attributes = (
            means.detach().cpu().numpy(),
            torch.zeros_like(means).detach().cpu().numpy(),
            f_dc.detach().cpu().contiguous().numpy(),
            f_rest.detach().cpu().contiguous().numpy(),
            inverse_sigmoid(opacities)[..., None].detach().cpu().numpy(),
            scales.log().detach().cpu().numpy(),
            rotations,
        )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
