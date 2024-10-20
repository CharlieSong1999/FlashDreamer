# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from networks.layers import upsample, ConvBlock, Conv3x3

from einops import rearrange


class DepthDecoder(nn.Module):
    def __init__(self, cfg, num_ch_enc, num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.cfg = cfg
        depth_num = cfg.model.gaussians_per_pixel - 1 if "unidepth" in cfg.model.name else cfg.model.gaussians_per_pixel
        self.num_output_channels = num_output_channels * depth_num
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = cfg.model.scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

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

        for s in self.scales:
            out = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            self.convs[("dispconv", s)] = out
            nn.init.xavier_uniform_(out.conv.weight, cfg.model.depth_scale)
            nn.init.constant_(out.conv.bias, cfg.model.depth_bias)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        if cfg.model.depth_type in ["disp", "disp_inc"]:
            self.activate = nn.Sigmoid()
        elif cfg.model.depth_type == "depth":
            self.activate = nn.Softplus()
        elif cfg.model.depth_type == "depth_inc":
            self.activate = torch.exp

    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                depth_num = self.cfg.model.gaussians_per_pixel - 1 if "unidepth" in self.cfg.model.name else self.cfg.model.gaussians_per_pixel
                if self.cfg.model.depth_type == "depth_inc":
                    outputs[("depth", i)] = rearrange(self.activate(torch.clamp(self.convs[("dispconv", i)](x), min=-10.0, max=6.0)),
                                                 'b (n c) ...-> (b n) c ...', n = depth_num)
                elif self.cfg.model.depth_type in ["disp", "disp_inc"]:
                    outputs[("disp", i)] = rearrange(self.activate(self.convs[("dispconv", i)](x)),
                                                 'b (n c) ...-> (b n) c ...', n = depth_num)
                else:
                    outputs[(self.cfg.model.depth_type, i)] = rearrange(self.activate(self.convs[("dispconv", i)](x)),
                                                 'b (n c) ...-> (b n) c ...', n = depth_num)
        return outputs
