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


def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, "gaussian 1"],
    path: Path,
):
    path = Path(path)

    min_opacity = 0.50
    valid = opacities[..., 0] > min_opacity
    means = means[valid, ...]
    scales = scales[valid, ...]
    rotations = rotations[valid, ...]
    harmonics = harmonics[valid, ...]
    opacities = opacities[valid, ...]

    # Shift the scene so that the median Gaussian is at the origin.
    means = means - means.median(dim=0).values

    # Rescale the scene so that most Gaussians are within range [-1, 1].
    scale_factor = means.abs().quantile(0.95, dim=0).max()
    means = means / scale_factor
    scales = scales / scale_factor
    scales = scales * 4.0
    scales = torch.clamp(scales, 0, 0.0075)

    # Define a rotation that makes +Z be the world up vector.
    # rotation = [
    #     [0, 0, 1],
    #     [-1, 0, 0],
    #     [0, -1, 0],
    # ]
    rotation = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    rotation = torch.tensor(rotation, dtype=torch.float32, device=means.device)

    # The Polycam viewer seems to start at a 45 degree angle. Since we want to be
    # looking directly at the object, we compose a 45 degree rotation onto the above
    # rotation.
    # adjustment = torch.tensor(
    #     R.from_rotvec([0, 0, -45], True).as_matrix(),
    #     dtype=torch.float32,
    #     device=means.device,
    # )
    # rotation = adjustment @ rotation

    # We also want to see the scene in camera space (as the default view). We therefore
    # compose the w2c rotation onto the above rotation.
    # rotation = rotation @ extrinsics[:3, :3].inverse()

    # Apply the rotation to the means (Gaussian positions).
    means = einsum(rotation, means, "i j, ... j -> ... i")

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = rotation.detach().cpu().numpy() @ rotations
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since our axes are swizzled for the spherical harmonics, we only export the DC
    # band.
    harmonics_view_invariant = harmonics

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        harmonics_view_invariant.detach().cpu().contiguous().numpy(),
        opacities.detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    # print(elements)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
    # save pt
    save_path = '/home/wjq/24s2/week2/test/output/tensors.pt'
    torch.save({
        'means': means,
        'scales': scales,
        'rotations': rotations,
        'harmonics': harmonics,
        'opacities': opacities
    }, save_path)


def save_ply(outputs, path, num_gauss=3, h=256, w=384, pad=32):
    print(h, w)
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

    export_ply(
        means,
        scales,
        rotations,
        harmonics,
        opacities,
        path
    )