from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .unidepth import UniDepthDepth
from .resnet_encoder import ResnetEncoder
from .gaussian_decoder import GaussianDecoder
from .depth_decoder import DepthDecoder

from networks.layers import disp_to_depth
from networks.gaussian_decoder import get_splits_and_inits


class UniDepthExtended(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.cfg = cfg

        self.unidepth = UniDepthDepth(cfg)

        self.parameters_to_train = []
        if self.cfg.model.splat_branch == "resnet":
            self.encoder = ResnetEncoder(cfg.model.num_layers,
                                         cfg.model.weights_init == "pretrained",
                                         cfg.model.resnet_bn_order
                                        )
            # change encoder to take depth as conditioning
            if self.cfg.model.depth_cond:
                self.encoder.encoder.conv1 = nn.Conv2d(
                    4,
                    self.encoder.encoder.conv1.out_channels,
                    kernel_size = self.encoder.encoder.conv1.kernel_size,
                    padding = self.encoder.encoder.conv1.padding,
                    stride = self.encoder.encoder.conv1.stride
                )
            self.parameters_to_train += [{"params": self.encoder.parameters()}]

            # use depth branch only for more gaussians
            if cfg.model.gaussians_per_pixel > 1:
                models ={}
                models["depth"] = DepthDecoder(cfg, self.encoder.num_ch_enc)
                self.parameters_to_train +=[{"params": models["depth"].parameters()}]
                for i in range(cfg.model.gaussians_per_pixel):
                    models["gauss_decoder_"+str(i)] = GaussianDecoder(cfg, self.encoder.num_ch_enc)
                    self.parameters_to_train += [{"params": models["gauss_decoder_"+str(i)].parameters()}]
                    if cfg.model.one_gauss_decoder:
                        break
                self.models = nn.ModuleDict(models)
            else:
                self.gauss_decoder = GaussianDecoder(cfg, self.encoder.num_ch_enc)
                self.parameters_to_train += [{"params": self.gauss_decoder.parameters()}]
        
        elif self.cfg.model.splat_branch == "unidepth_vit" or self.cfg.model.splat_branch == "unidepth_cnvnxtl":
            self.splat_branch = UniDepthDepth(cfg,
                                              return_raw_preds=True)
            # modify the head to output the channels for Gaussian parameters
            self.init_ouput_head_splat_branch()
            self.parameters_to_train +=[{"params": self.splat_branch.parameters()}]

        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def init_ouput_head_splat_branch(self):
        split_dimensions, scale, bias = get_splits_and_inits(self.cfg)
        # the first dim in the output is for depth - we don't use that in this branch
        self.split_dimensions = split_dimensions[1:]
        scale = scale[1:]
        bias = bias[1:]

        self.num_output_channels = sum(self.split_dimensions)

        self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out2 = \
            nn.Conv2d(self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out2.in_channels, 
                      self.num_output_channels,
                kernel_size = self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out2.kernel_size,
                padding = self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out2.padding)

        self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out4 = \
            nn.Conv2d(self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out4.in_channels, 
                      self.num_output_channels,
                kernel_size = self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out4.kernel_size,
                padding = self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out4.padding)

        self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out8 = \
            nn.Conv2d(self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out8.in_channels, 
                      self.num_output_channels,
                kernel_size = self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out8.kernel_size,
                padding = self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out8.padding)

        start_channels = 0
        for out_channel, b, s in zip(split_dimensions, bias, scale):
            nn.init.xavier_uniform_(
                self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out2.weight[start_channels:start_channels+out_channel,
                                :, :, :], s)
            nn.init.constant_(
                self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out2.bias[start_channels:start_channels+out_channel], b)
            start_channels += out_channel
        
        start_channels = 0
        for out_channel, b, s in zip(split_dimensions, bias, scale):
            nn.init.xavier_uniform_(
                self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out4.weight[start_channels:start_channels+out_channel,
                                :, :, :], s)
            nn.init.constant_(
                self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out4.bias[start_channels:start_channels+out_channel], b)
            start_channels += out_channel

        start_channels = 0
        for out_channel, b, s in zip(split_dimensions, bias, scale):
            nn.init.xavier_uniform_(
                self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out8.weight[start_channels:start_channels+out_channel,
                                :, :, :], s)
            nn.init.constant_(
                self.splat_branch.depth_prediction_model.pixel_decoder.depth_layer.out8.bias[start_channels:start_channels+out_channel], b)
            start_channels += out_channel

    def get_parameter_groups(self):
        # only the resnet encoder and gaussian parameter decoder are optimisable
        return self.parameters_to_train

    def forward(self, inputs):
        if ('unidepth', 0, 0) in inputs.keys() and inputs[('unidepth', 0, 0)] is not None:
            depth_outs = dict()
            depth_outs["depth"] = inputs[('unidepth', 0, 0)]
        else:
            with torch.no_grad():
                _, depth_outs = self.unidepth(inputs)

        outputs_gauss = {}

        K = depth_outs["intrinsics"]
        outputs_gauss[("K_src", 0)] = K
        outputs_gauss[("inv_K_src", 0)] = torch.linalg.inv(K)

        if self.cfg.model.splat_branch == "resnet":
            if self.cfg.model.depth_cond:
                # division by 20 is to put depth in a similar range to RGB
                resnet_input = torch.cat([inputs["color_aug", 0, 0], 
                                          depth_outs["depth"] / 20.0], dim=1)
            else:
                resnet_input = inputs["color_aug", 0, 0]
            resnet_features = self.encoder(resnet_input)
            if self.cfg.model.gaussians_per_pixel > 1:
                pred_depth = dict()
                depth = self.models["depth"](resnet_features)
                if self.cfg.model.depth_type == "disp":
                    for key, v in depth.items():
                        _, pred_depth[("depth", key[1])] = disp_to_depth(v, self.cfg.model.min_depth, self.cfg.model.max_depth)
                elif self.cfg.model.depth_type in ["depth", "depth_inc"]:
                    pred_depth = depth
                pred_depth[("depth", 0)] = rearrange(pred_depth[("depth", 0)], "(b n) ... -> b n ...", n=self.cfg.model.gaussians_per_pixel - 1)
                if self.cfg.model.depth_type in ["depth_inc", "disp_inc"]:
                    pred_depth[("depth", 0)] = torch.cumsum(torch.cat((depth_outs["depth"][:,None,...], pred_depth[("depth", 0)]), dim=1), dim=1)
                else:
                    pred_depth[("depth", 0)] = torch.cat((depth_outs["depth"][:,None,...], pred_depth[("depth", 0)]), dim=1)
                outputs_gauss[("depth", 0)] = rearrange(pred_depth[("depth", 0)], "b n c ... -> (b n) c ...", n = self.cfg.model.gaussians_per_pixel)
                gauss_outs = dict()
                for i in range(self.cfg.model.gaussians_per_pixel):
                    outs = self.models["gauss_decoder_"+str(i)](resnet_features)
                    if not self.cfg.model.one_gauss_decoder:
                        for key, v in outs.items():
                            gauss_outs[key] = outs[key][:,None,...] if i==0 else torch.cat([gauss_outs[key], outs[key][:,None,...]], dim=1)
                    else:
                        gauss_outs |= outs
                for key, v in gauss_outs.items():
                    gauss_outs[key] = rearrange(gauss_outs[key], 'b n ... -> (b n) ...')
                outputs_gauss |= gauss_outs
            else:
                outputs_gauss[("depth", 0)] = depth_outs["depth"]
                outputs_gauss |= self.gauss_decoder(resnet_features)
        elif self.cfg.model.splat_branch == "unidepth_vit" or self.cfg.model.splat_branch == "unidepth_cnvnxtl":
            split_network_outputs = self.splat_branch(inputs)[1].split(self.split_dimensions, dim=1)
            offset, opacity, scaling, rotation, feat_dc = split_network_outputs[:5]

            outputs_gauss |= {
                ("gauss_opacity", 0): self.opacity_activation(opacity),
                ("gauss_scaling", 0): self.scaling_activation(scaling),
                ("gauss_rotation", 0): self.rotation_activation(rotation),
                ("gauss_features_dc", 0): feat_dc
            }

            if self.cfg.model.max_sh_degree > 0:
                features_rest = split_network_outputs[5]
                outputs_gauss[("gauss_features_rest", 0)] = features_rest

            assert self.cfg.model.predict_offset
            outputs_gauss[("gauss_offset", 0)] = offset

        return outputs_gauss
    
