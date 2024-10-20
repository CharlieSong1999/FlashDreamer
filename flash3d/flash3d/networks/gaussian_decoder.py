<<<<<<< HEAD
from __future__ import annotations
=======
>>>>>>> 2e788b3 (commit)
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


def get_splits_and_inits(cfg):
    split_dimensions = []
    scale_inits = []
    bias_inits = []

    for g_idx in range(cfg.model.gaussians_per_pixel):
        if cfg.model.predict_offset:
            split_dimensions += [3]
            scale_inits += [cfg.model.xyz_scale]
            bias_inits += [cfg.model.xyz_bias]

        split_dimensions += [1, 3, 4, 3]
        scale_inits += [cfg.model.opacity_scale, 
                        cfg.model.scale_scale,
                        1.0,
                        5.0]
        bias_inits += [cfg.model.opacity_bias,
                        np.log(cfg.model.scale_bias),
                        0.0,
                        0.0]

        if cfg.model.max_sh_degree != 0:
            sh_num = (cfg.model.max_sh_degree + 1) ** 2 - 1
            sh_num_rgb = sh_num * 3
            split_dimensions.append(sh_num_rgb)
            scale_inits.append(cfg.model.sh_scale)
            bias_inits.append(0.0)
        if not cfg.model.one_gauss_decoder:
            break

    return split_dimensions, scale_inits, bias_inits, 


class GaussianDecoder(nn.Module):
    def __init__(self, cfg, num_ch_enc, use_skips=True):
        super(GaussianDecoder, self).__init__()

        self.cfg = cfg
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array(cfg.model.num_ch_dec)

        split_dimensions, scale, bias = get_splits_and_inits(cfg)

        # [offset], opacity, scaling, rotation, feat_dc
        assert not cfg.model.unified_decoder

        self.split_dimensions = split_dimensions

        self.num_output_channels = sum(self.split_dimensions)

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        self.out = nn.Conv2d(self.num_ch_dec[0], self.num_output_channels, 1)

        out_channels = self.split_dimensions
        start_channels = 0
        for out_channel, b, s in zip(out_channels, bias, scale):
            nn.init.xavier_uniform_(
                self.out.weight[start_channels:start_channels+out_channel,
                                :, :, :], s)
            nn.init.constant_(
                self.out.bias[start_channels:start_channels+out_channel], b)
            start_channels += out_channel

        self.decoder = nn.ModuleList(list(self.convs.values()))

        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.scaling_lambda = cfg.model.scale_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)

        x = self.out(x)

        split_network_outputs = x.split(self.split_dimensions, dim=1)

        offset_list = []
        opacity_list = []
        scaling_list = []
        rotation_list = []
        feat_dc_list = []
        feat_rest_list = []

        assert not self.cfg.model.unified_decoder

        for i in range(self.cfg.model.gaussians_per_pixel):
            assert self.cfg.model.max_sh_degree > 0
            if self.cfg.model.predict_offset:
                offset_s, opacity_s, scaling_s, \
                    rotation_s, feat_dc_s, features_rest_s = split_network_outputs[i*6:(i+1)*6]
                offset_list.append(offset_s[:, None, ...])
            else:
                opacity_s, scaling_s, rotation_s, feat_dc_s, features_rest_s = split_network_outputs[i*5:(i+1)*5]
            opacity_list.append(opacity_s[:, None, ...])
            scaling_list.append(scaling_s[:, None, ...])
            rotation_list.append(rotation_s[:, None, ...])
            feat_dc_list.append(feat_dc_s[:, None, ...])
            feat_rest_list.append(features_rest_s[:, None, ...])
            if not self.cfg.model.one_gauss_decoder:
                break

        # squeezing will remove dimension if there is only one gaussian per pixel
        opacity = torch.cat(opacity_list, dim=1).squeeze(1)
        scaling = torch.cat(scaling_list, dim=1).squeeze(1)
        rotation = torch.cat(rotation_list, dim=1).squeeze(1)
        feat_dc = torch.cat(feat_dc_list, dim=1).squeeze(1)
        features_rest = torch.cat(feat_rest_list, dim=1).squeeze(1)

        out = {
            ("gauss_opacity", 0): self.opacity_activation(opacity),
            ("gauss_scaling", 0): self.scaling_activation(scaling) * self.scaling_lambda,
            ("gauss_rotation", 0): self.rotation_activation(rotation),
            ("gauss_features_dc", 0): feat_dc,
            ("gauss_features_rest", 0): features_rest
        }

        if self.cfg.model.predict_offset:
            offset = torch.cat(offset_list, dim=1).squeeze(1)
            out[("gauss_offset", 0)] = offset
        return out

