from pathlib import Path
from jaxtyping import Float
import numpy as np
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement
import torch
from torch import Tensor
from einops import rearrange, einsum


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

def postprocess(outputs, num_gauss=2, h=256, w=384, pad=32):

    def crop_r(t):
        H = h + pad * 2
        W = w + pad * 2
        t = rearrange(t, "b c (h w) -> b c h w", h=H, w=W)
        t = t[..., pad:H-pad, pad:W-pad]
        t = rearrange(t, "b c h w -> b c (h w)")
        return t

    def crop(t):
        H = h + pad * 2
        W = w + pad * 2
        t = t[..., pad:H-pad, pad:W-pad]
        return t
    means = rearrange(crop_r(outputs[('gauss_means', 0, 0)]), "(b v) c n -> b (v n) c", v=num_gauss)[0, :, :3]
    scales = rearrange(crop(outputs[('gauss_scaling', 0, 0)]), "(b v) c h w -> b (v h w) c", v=num_gauss)[0]
    rotations = rearrange(crop(outputs[('gauss_rotation', 0, 0)]), "(b v) c h w -> b (v h w) c", v=num_gauss)[0]
    opacities = rearrange(crop(outputs[('gauss_opacity', 0, 0)]), "(b v) c h w -> b (v h w) c", v=num_gauss)[0]
    harmonics = rearrange(crop(outputs[('gauss_features_dc', 0, 0)]), "(b v) c h w -> b (v h w) c", v=num_gauss)[0]
    
    # 恢复场景尺度
    K_input = torch.tensor([
                    [600.0/2, 0, 599.5/2],
                    [0, 600.0/2, 339.5/2],
                    [0, 0, 1]
                ])
    K_output = outputs[('K_src', 0, 0)].squeeze(0)

    scale_x = K_input[0, 0] / K_output[0, 0]  # fx1 / fx2
    scale_y = K_input[1, 1] / K_output[1, 1]  # fy1 / fy2
    
    means[:, 0] /= scale_x
    means[:, 1] /= scale_y

    cx_diff = (K_input[0, 2] - K_output[0, 2]) / K_input[0, 0]
    cy_diff = (K_input[1, 2] - K_output[1, 2]) / K_input[1, 1]
    
    means[:, 0] -= cx_diff * means[:, 2]
    means[:, 1] -= cy_diff * means[:, 2]

    return export_ply(
            means,
            scales,
            rotations,
            harmonics,
            opacities,
        )

def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, "gaussian 1"],
):

    rotation = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    rotation = torch.tensor(rotation, dtype=torch.float32, device=means.device)

    # Apply the rotation to the means (Gaussian positions).
    # means = einsum(rotation, means, "i j, ... j -> ... i")

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = rotation.detach().cpu().numpy() @ rotations
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    result = {
        'means': means,
        'scales': scales,
        'rotations': rotations,
        'harmonics': harmonics,
        'opacities': opacities
    }

    return result

def save_ply(params, path):

    path = Path(path)
    means = params['means']
    harmonics_view_invariant = params['harmonics']
    opacities = params['opacities']
    scales = params['scales']
    rotations = params['rotations']

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        harmonics_view_invariant.detach().cpu().contiguous().numpy(),
        opacities.detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations.detach().cpu().numpy(),
    )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)

