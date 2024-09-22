from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange

from unidepth.models.unidepthv1 import UniDepthV1
from unidepth.utils.constants import IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD
from unidepth.utils.geometric import (
    generate_rays,
    spherical_zbuffer_to_euclidean,
    flat_interpolate,
)
from unidepth.layers import (
    MLP,
    AttentionBlock,
    NystromBlock,
    PositionEmbeddingSine,
    ConvUpsample,
)
from unidepth.utils.sht import rsh_cart_8

from networks.gaussian_decoder import get_splits_and_inits


# inference helpers
def _paddings(image_shape, network_shape):
    cur_h, cur_w = image_shape
    h, w = network_shape
    pad_top, pad_bottom = (h - cur_h) // 2, h - cur_h - (h - cur_h) // 2
    pad_left, pad_right = (w - cur_w) // 2, w - cur_w - (w - cur_w) // 2
    return pad_left, pad_right, pad_top, pad_bottom


def _shapes(image_shape, network_shape):
    h, w = image_shape
    input_ratio = w / h
    output_ratio = network_shape[1] / network_shape[0]
    if output_ratio > input_ratio:
        ratio = network_shape[0] / h
    elif output_ratio <= input_ratio:
        ratio = network_shape[1] / w
    return (ceil(h * ratio - 0.5), ceil(w * ratio - 0.5)), ratio


def _preprocess(rgbs, intrinsics, shapes, pads, ratio, output_shapes):
    (pad_left, pad_right, pad_top, pad_bottom) = pads
    rgbs = F.interpolate(
        rgbs, size=shapes, mode="bilinear", align_corners=False, antialias=True
    )
    rgbs = F.pad(rgbs, (pad_left, pad_right, pad_top, pad_bottom), mode="constant")
    if intrinsics is not None:
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * ratio
        intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * ratio
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * ratio + pad_left
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * ratio + pad_top
        return rgbs, intrinsics
    return rgbs, None


def _postprocess(predictions, intrinsics, shapes, pads, ratio, original_shapes):
    
    (pad_left, pad_right, pad_top, pad_bottom) = pads
    # pred mean, trim paddings, and upsample to input dim
    predictions = sum(
        [
            F.interpolate(
                x,
                size=shapes,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            for x in predictions
        ]
    ) / len(predictions)

    shapes = predictions.shape[2:]
    predictions = predictions[
        ..., pad_top : shapes[0] - pad_bottom, pad_left : shapes[1] - pad_right
    ]

    predictions = F.interpolate(
        predictions,
        size=original_shapes,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    if intrinsics is not None:
        intrinsics[:, 0, 0] = intrinsics[:, 0, 0] / ratio
        intrinsics[:, 1, 1] = intrinsics[:, 1, 1] / ratio
        intrinsics[:, 0, 2] = (intrinsics[:, 0, 2] - pad_left) / ratio
        intrinsics[:, 1, 2] = (intrinsics[:, 1, 2] - pad_top) / ratio

    return predictions, intrinsics


def scale_intrinsics_xy(intrinsics, x_ratio, y_ratio):
    intrinsics = intrinsics.clone()
    intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * x_ratio
    intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * y_ratio
    intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * x_ratio
    intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * y_ratio
    return intrinsics


def scale_intrinsics(intrinsics, ratio):
    intrinsics = intrinsics.clone()
    intrinsics[:, 0, 0] = intrinsics[:, 0, 0] * ratio
    intrinsics[:, 1, 1] = intrinsics[:, 1, 1] * ratio
    intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * ratio
    intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * ratio
    return intrinsics


def unidepthv1_forward(model, rgbs, intrinsics, skip_camera,
                       return_raw_preds=False):
    B, _, H, W = rgbs.shape

    rgbs = TF.normalize(
        rgbs,
        mean=IMAGENET_DATASET_MEAN,
        std=IMAGENET_DATASET_STD,
    )

    (h, w), ratio = _shapes((H, W), model.image_shape)
    pad_left, pad_right, pad_top, pad_bottom = _paddings((h, w), model.image_shape)
    rgbs, gt_intrinsics = _preprocess(
        rgbs,
        intrinsics,
        (h, w),
        (pad_left, pad_right, pad_top, pad_bottom),
        ratio,
        model.image_shape,
    )
    
    encoder_outputs, cls_tokens = model.pixel_encoder(rgbs)
    if "dino" in model.pixel_encoder.__class__.__name__.lower():
        encoder_outputs = [
            (x + y.unsqueeze(1)).contiguous()
            for x, y in zip(encoder_outputs, cls_tokens)
        ]
    
    # get data for decoder and adapt to given camera
    inputs = {}
    inputs["encoder_outputs"] = encoder_outputs
    inputs["cls_tokens"] = cls_tokens
    inputs["image"] = rgbs
    if gt_intrinsics is not None:
        rays, angles = generate_rays(
            gt_intrinsics, model.image_shape, noisy=False
        )
        inputs["rays"] = rays
        inputs["angles"] = angles
        inputs["K"] = gt_intrinsics
        model.pixel_decoder.test_fixed_camera = True
        model.pixel_decoder.skip_camera = skip_camera

    # decode all
    pred_intrinsics, predictions, features, rays = model.pixel_decoder(inputs, {})

    pads = (pad_left, pad_right, pad_top, pad_bottom)

    # undo the reshaping and get original image size (slow)
    predictions, pred_intrinsics = _postprocess(
        predictions,
        pred_intrinsics,
        model.image_shape,
        pads,
        ratio,
        (H, W),
    )

    if return_raw_preds:
        return inputs, predictions

    # final 3D points backprojection
    intrinsics = gt_intrinsics if gt_intrinsics is not None else pred_intrinsics
    angles = generate_rays(intrinsics, (H, W), noisy=False)[-1]
    angles = rearrange(angles, "b (h w) c -> b c h w", h=H, w=W)
    points_3d = torch.cat((angles, predictions), dim=1)
    points_3d = spherical_zbuffer_to_euclidean(
        points_3d.permute(0, 2, 3, 1)
    ).permute(0, 3, 1, 2)

    # output data
    outputs = {
        "intrinsics": intrinsics,
        "points": points_3d,
        "depth": predictions[:, -1:],
        "depth_feats": features,
        "rays": rays,
        "padding": pads
    }
    model.pixel_decoder.test_fixed_camera = False
    model.pixel_decoder.skip_camera = False
    return inputs, outputs

class UniDepthDepth(nn.Module):
    def __init__(
        self,
        cfg,
        return_raw_preds=False
    ):
        super().__init__()

        self.cfg = cfg
        self.return_raw_preds = return_raw_preds

        if "cnvnxtl" in cfg.model.name:
            self.depth_prediction_model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-cnvnxtl")
        elif "vit" in cfg.model.name:
            self.depth_prediction_model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")

        self.skip_camera = True

    def get_depth(self, img, intrinsics):
        depth_inputs, outputs = unidepthv1_forward(
            self.depth_prediction_model, 
            img, 
            intrinsics, 
            self.skip_camera,
            return_raw_preds=self.return_raw_preds)
        return outputs

    def forward(self, inputs):
        input_img = inputs["color_aug", 0, 0]
        # here we need the intrinsics of the source image to condition on
        # the depth prediction. needs to account for padding 
        if ("K_src", 0) in inputs:
            intrinsics = inputs[("K_src", 0)]
        else:
            intrinsics = None

        depth_inputs, outputs = unidepthv1_forward(
            self.depth_prediction_model, 
            input_img, 
            intrinsics, 
            self.skip_camera,
            return_raw_preds=self.return_raw_preds)

        return depth_inputs, outputs

class UniDepthUnprojector(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.cfg = cfg

        if cfg.model.name == "unidepth_unprojector_cnvnxtl":
            model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-cnvnxtl")
        elif cfg.model.name == "unidepth_unprojector_vit":
            model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
        self.unidepth = model

        self.skip_camera = True

        self.register_buffer("gauss_opacity", torch.ones(1, 1, 1).float())
        self.register_buffer("gauss_scaling", torch.ones(3, 1, 1).float())
        self.register_buffer("gauss_rotation", torch.ones(4, 1, 1).float() * 0.5)
        self.register_buffer("gauss_features_rest", torch.zeros(9, 1, 1).float())
        self.register_buffer("gauss_offset", torch.zeros(3, 1, 1).float())

        self.all_params = nn.ParameterDict({
                           "opacity_scaling": nn.Parameter(torch.tensor(cfg.model.opacity_bias).float()),
                           "scale_scaling": nn.Parameter(torch.tensor(cfg.model.scale_bias).float()),
                           "colour_scaling": nn.Parameter(torch.tensor(self.cfg.model.colour_scale).float())})

        
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.relu = nn.ReLU()

    def get_parameter_groups(self):
        # tune scalars for size, opacity and colour modulation
        return [{'params': self.all_params.parameters()}]

    def forward(self, inputs):
        model = self.unidepth
        input_img = inputs["color_aug", 0, 0]
        # here we need the intrinsics of the source image to condition on
        # the depth prediction. needs to account for padding 
        intrinsics = inputs[("K_src", 0)]
        b, c, h, w = inputs["color_aug", 0, 0].shape

        with torch.no_grad():
            _, depth_outs = unidepthv1_forward(model, input_img, intrinsics, self.skip_camera)

        outs = {}

        outs[("gauss_opacity", 0)] = self.gauss_opacity.unsqueeze(0).expand(depth_outs["depth"].shape[0], -1, h, w) \
            * self.opacity_activation(self.all_params["opacity_scaling"])
        if not self.cfg.model.scale_with_depth:
            outs[("gauss_scaling", 0)] = self.gauss_scaling.unsqueeze(0).expand(depth_outs["depth"].shape[0], -1, h, w) \
                * self.scaling_activation(self.all_params["scale_scaling"])
        else:
            outs[("gauss_scaling", 0)] = self.gauss_scaling.unsqueeze(0).expand(depth_outs["depth"].shape[0], -1, h, w) \
                * self.scaling_activation(self.all_params["scale_scaling"]) * depth_outs["depth"] / 10.0
        outs[("gauss_rotation", 0)] = self.gauss_rotation.unsqueeze(0).expand(depth_outs["depth"].shape[0], -1, h, w)
        outs[("gauss_offset", 0)] = self.gauss_offset.unsqueeze(0).expand(depth_outs["depth"].shape[0], -1, h, w)
        outs[("gauss_features_rest", 0)] = self.gauss_features_rest.unsqueeze(0).expand(depth_outs["depth"].shape[0], -1, h, w)
        # rendering adds 0.5 to go from rendered colours to output
        outs[("gauss_features_dc", 0)] = (input_img - 0.5)* self.relu(self.all_params["colour_scaling"])

        outs[("depth", 0)] = depth_outs["depth"]

        return outs

class UniDepthSplatter(nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.cfg = cfg

        config_path = Path("/work/eldar/src/UniDepth")
        with open(config_path / "configs/config_v1_cnvnxtl.json") as f:
            config = json.load(f)
        self.unidepth = UniDepthDepth(self.cfg)

        hidden_dim = config["model"]["pixel_decoder"]["hidden_dim"]
        expansion = config["model"]["expansion"]
        depth = config["model"]["pixel_decoder"]["depths"]
        num_heads = config["model"]["num_heads"]
        dropout = config["model"]["pixel_decoder"]["dropout"]
        layer_scale = 1.0
        self.splat_decoder = GaussSplatHead(
            cfg,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            depths=depth,
            camera_dim=81,
            dropout=dropout,
            layer_scale=layer_scale,
        )

        self.skip_camera = True

    def get_parameter_groups(self):
        base_lr = self.cfg.optimiser.learning_rate
        return [
            {'params': self.unidepth.parameters(), "lr": base_lr * 0.05},
            {'params': self.splat_decoder.parameters()}
        ]

    def forward(self, inputs):
        gauss_head = self.splat_decoder

        depth_inputs, depth_outs = self.unidepth(inputs)
        depth_feats = depth_outs["depth_feats"]
        rays = depth_outs["rays"]
        padding = depth_outs["padding"]
        
        B, _, H, W = depth_inputs["image"].shape

        # TODO remove hardcoded shapes
        common_shape = (28, 38)
        gauss_head.set_shapes(common_shape)
        gauss_head.set_original_shapes((H, W))

        depth_feats = rearrange(depth_feats, "b c h w -> b (h w) c")
        outs = gauss_head(
            latents_16=depth_feats,
            rays_hr=rays,
        )
        for k, v in outs.items():
            pred, _ = _postprocess([v], None, self.unidepth.depth_prediction_model.image_shape, 
                                   padding, None, inputs["color_aug", 0, 0].shape[2:4])
            outs[k] = pred
        outs[("depth", 0)] = depth_outs["depth"]

        return outs


class GaussSplatHead(nn.Module):
    def __init__(
        self,
        cfg,
        hidden_dim: int,
        num_heads: int = 8,
        expansion: int = 4,
        depths: int | list[int] = 4,
        camera_dim: int = 256,
        dropout: float = 0.0,
        layer_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.cfg = cfg

        if isinstance(depths, int):
            depths = [depths] * 3
        assert len(depths) == 3

        self.project_rays16 = MLP(
            camera_dim, expansion=expansion, dropout=dropout, output_dim=hidden_dim
        )
        self.project_rays8 = MLP(
            camera_dim, expansion=expansion, dropout=dropout, output_dim=hidden_dim // 2
        )
        self.project_rays4 = MLP(
            camera_dim, expansion=expansion, dropout=dropout, output_dim=hidden_dim // 4
        )

        self.layers_8 = nn.ModuleList([])
        self.layers_4 = nn.ModuleList([])
        layers_16 = nn.ModuleList([])

        self.up8 = ConvUpsample(
            hidden_dim, expansion=expansion, layer_scale=layer_scale
        )
        self.up4 = ConvUpsample(
            hidden_dim // 2, expansion=expansion, layer_scale=layer_scale
        )
        self.up2 = ConvUpsample(
            hidden_dim // 4, expansion=expansion, layer_scale=layer_scale
        )

        split_dimensions, scale, bias = get_splits_and_inits(cfg)
        start = 1
        self.split_dimensions = split_dimensions[start:]
        scale = scale[start:]
        bias = bias[start:]

        self.num_output_channels = sum(self.split_dimensions)

        self.out2 = nn.Conv2d(hidden_dim // 8, self.num_output_channels, 3, padding=1)
        # self.out4 = nn.Conv2d(hidden_dim // 4, self.num_output_channels, 3, padding=1)
        # self.out8 = nn.Conv2d(hidden_dim // 2, self.num_output_channels, 3, padding=1)

        start_channels = 0
        for out_channel, b, s in zip(self.split_dimensions, bias, scale):
            nn.init.xavier_uniform_(
                self.out2.weight[start_channels:start_channels+out_channel,
                                :, :, :], s)
            nn.init.constant_(
                self.out2.bias[start_channels:start_channels+out_channel], b)
            start_channels += out_channel

        for i, (blk_lst, depth) in enumerate(
            zip([layers_16, self.layers_8, self.layers_4], depths)
        ):
            if i == 0:
                continue
            attn_cls = AttentionBlock if i == 0 else NystromBlock
            for _ in range(depth):
                blk_lst.append(
                    attn_cls(
                        hidden_dim // (2**i),
                        num_heads=num_heads // (2**i),
                        expansion=expansion,
                        dropout=dropout,
                        layer_scale=layer_scale,
                    )
                )

        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.scaling_lambda = cfg.model.scale_lambda
        self.sigmoid = nn.Sigmoid()

    def set_original_shapes(self, shapes: Tuple[int, int]):
        self.original_shapes = shapes

    def set_shapes(self, shapes: Tuple[int, int]):
        self.shapes = shapes

    def forward(
        self, latents_16: torch.Tensor, rays_hr: torch.Tensor
    ) -> torch.Tensor:
        shapes = self.shapes

        # camera_embedding
        # torch.cuda.synchronize()
        # start = time()
        rays_embedding_16 = F.normalize(
            flat_interpolate(rays_hr, old=self.original_shapes, new=shapes), dim=-1
        )
        rays_embedding_8 = F.normalize(
            flat_interpolate(
                rays_hr, old=self.original_shapes, new=[x * 2 for x in shapes]
            ),
            dim=-1,
        )
        rays_embedding_4 = F.normalize(
            flat_interpolate(
                rays_hr, old=self.original_shapes, new=[x * 4 for x in shapes]
            ),
            dim=-1,
        )
        rays_embedding_16 = self.project_rays16(rsh_cart_8(rays_embedding_16))
        rays_embedding_8 = self.project_rays8(rsh_cart_8(rays_embedding_8))
        rays_embedding_4 = self.project_rays4(rsh_cart_8(rays_embedding_4))

        # Block 16 - Out 8
        latents_8 = self.up8(
            rearrange(
                latents_16 + rays_embedding_16,
                "b (h w) c -> b c h w",
                h=shapes[0],
                w=shapes[1],
            ).contiguous()
        )
        # out8 = self.out8(
        #     rearrange(
        #         latents_8, "b (h w) c -> b c h w", h=shapes[0] * 2, w=shapes[1] * 2
        #     )
        # )

        # Block 8 - Out 4
        for layer in self.layers_8:
            latents_8 = layer(latents_8, pos_embed=rays_embedding_8)
        latents_4 = self.up4(
            rearrange(
                latents_8 + rays_embedding_8,
                "b (h w) c -> b c h w",
                h=shapes[0] * 2,
                w=shapes[1] * 2,
            ).contiguous()
        )
        # out4 = self.out4(
        #     rearrange(
        #         latents_4, "b (h w) c -> b c h w", h=shapes[0] * 4, w=shapes[1] * 4
        #     )
        # )

        # Block 4 - Out 2
        for layer in self.layers_4:
            latents_4 = layer(latents_4, pos_embed=rays_embedding_4)
        latents_2 = self.up2(
            rearrange(
                latents_4 + rays_embedding_4,
                "b (h w) c -> b c h w",
                h=shapes[0] * 4,
                w=shapes[1] * 4,
            ).contiguous()
        )
        out2 = self.out2(
            rearrange(
                latents_2, "b (h w) c -> b c h w", h=shapes[0] * 8, w=shapes[1] * 8
            )
        )

        split_network_outputs = out2.split(self.split_dimensions, dim=1)
        last = 5
        offset, opacity, scaling, rotation, feat_dc = split_network_outputs[:last]

        out = {
            ("gauss_opacity", 0): self.opacity_activation(opacity),
            ("gauss_scaling", 0): self.scaling_activation(scaling) * self.scaling_lambda,
            ("gauss_rotation", 0): self.rotation_activation(rotation),
            ("gauss_features_dc", 0): feat_dc
        }

        if self.cfg.model.max_sh_degree > 0:
            features_rest = split_network_outputs[last]
            out[("gauss_features_rest", 0)] = features_rest

        if self.cfg.model.predict_offset:
            out[("gauss_offset", 0)] = offset

        return out
        # return out8, out4, out2, proj_latents_16
