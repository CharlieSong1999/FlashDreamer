<<<<<<< HEAD
from __future__ import annotations
=======
>>>>>>> 2e788b3 (commit)
from pathlib import Path
import logging

import torch
import torch.nn as nn
from einops import rearrange

from networks.layers import BackprojectDepth, disp_to_depth
from networks.resnet_encoder import ResnetEncoder
from networks.depth_decoder import DepthDecoder
from networks.gaussian_decoder import GaussianDecoder


def default_param_group(model):
    return [{'params': model.parameters()}]


def to_device(inputs, device):
    for key, ipt in inputs.items():
        if isinstance(ipt, torch.Tensor):
            inputs[key] = ipt.to(device)
    return inputs


class GaussianPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # checking height and width are multiples of 32
        # assert cfg.dataset.width % 32 == 0, "'width' must be a multiple of 32"

        models = {}
        self.parameters_to_train = []

        self.num_scales = len(cfg.model.scales)

        assert cfg.model.frame_ids[0] == 0, "frame_ids must start with 0"

        if cfg.model.use_stereo:
            cfg.model.frame_ids.append("s")

        model_name = cfg.model.name
        if model_name == "resnet":
            models["encoder"] = ResnetEncoder(
                cfg.model.num_layers,
                cfg.model.weights_init == "pretrained",
                cfg.model.resnet_bn_order
            )
            self.parameters_to_train += default_param_group(models["encoder"])
            if not cfg.model.unified_decoder:
                models["depth"] = DepthDecoder(
                    cfg, models["encoder"].num_ch_enc)
                self.parameters_to_train += default_param_group(models["depth"])
            if cfg.model.gaussian_rendering:
                for i in range(cfg.model.gaussians_per_pixel):
                    gauss_decoder = GaussianDecoder(
                        cfg, models["encoder"].num_ch_enc,
                    )
                    self.parameters_to_train += default_param_group(gauss_decoder)
                    models["gauss_decoder_"+str(i)] = gauss_decoder
        elif model_name == "unidepth":
            from networks.unidepth import UniDepthSplatter
            models["unidepth"] = UniDepthSplatter(cfg)
            self.parameters_to_train += models["unidepth"].get_parameter_groups()
        elif model_name in ["unidepth_unprojector_vit", "unidepth_unprojector_cnvnxtl"]:
            from networks.unidepth import UniDepthUnprojector
            models["unidepth"] = UniDepthUnprojector(cfg)
            self.parameters_to_train += models["unidepth"].get_parameter_groups()
        elif model_name in ["unidepth_extension_vit", "unidepth_extension_cnvnxtl"]:
            from networks.unidepth_extension import UniDepthExtended
            models["unidepth_extended"] = UniDepthExtended(cfg)
            self.parameters_to_train += models["unidepth_extended"].get_parameter_groups()

        self.models = nn.ModuleDict(models)
        self.set_backproject()

    def set_backproject(self):
        backproject_depth = {}
        H = self.cfg.dataset.height
        W = self.cfg.dataset.width
        for scale in self.cfg.model.scales:
            h = H // (2 ** scale)
            w = W // (2 ** scale)
            if self.cfg.model.shift_rays_half_pixel == "zero":
                shift_rays_half_pixel = 0
            elif self.cfg.model.shift_rays_half_pixel == "forward":
                shift_rays_half_pixel = 0.5
            elif self.cfg.model.shift_rays_half_pixel == "backward":
                shift_rays_half_pixel = -0.5
            else:
                raise NotImplementedError
            backproject_depth[str(scale)] = BackprojectDepth(
                self.cfg.optimiser.batch_size * self.cfg.model.gaussians_per_pixel, 
                # backprojection can be different if padding was used
                h + 2 * self.cfg.dataset.pad_border_aug, 
                w + 2 * self.cfg.dataset.pad_border_aug,
                shift_rays_half_pixel=shift_rays_half_pixel
            )
        self.backproject_depth = nn.ModuleDict(backproject_depth)

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()
        self._is_train = True

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()
        self._is_train = False
    
    def is_train(self):
        return self._is_train
    
    def forward(self, inputs):
        cfg = self.cfg
        B = cfg.optimiser.batch_size

        if cfg.model.name == "resnet":
            do_flip = self.is_train() and \
                    cfg.train.lazy_flip_augmentation and \
                    (torch.rand(1) > .5).item()
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            input_img = inputs["color_aug", 0, 0]
            if do_flip:
                input_img = torch.flip(input_img, dims=(-1, ))
            features = self.models["encoder"](input_img)
            if not cfg.model.unified_decoder:
                outputs = self.models["depth"](features)
            else:
                outputs = dict()
            
            if self.cfg.model.gaussian_rendering:
                # gauss_feats = self.models["gauss_encoder"](inputs["color_aug", 0, 0])
                input_f_id = 0
                gauss_feats = features
                gauss_outs = dict()
                for i in range(self.cfg.model.gaussians_per_pixel):
                    outs = self.models["gauss_decoder_"+str(i)](gauss_feats)
                    for key, v in outs.items():
                        gauss_outs[key] = outs[key][:,None,...] if i==0 else torch.cat([gauss_outs[key], outs[key][:,None,...]], dim=1)
                for key, v in gauss_outs.items():
                    gauss_outs[key] = rearrange(gauss_outs[key], 'b n ... -> (b n) ...')
                outputs |= gauss_outs
                outputs = {(key[0], input_f_id, key[1]): v for key, v in outputs.items()}
            else:
                for scale in cfg.model.scales:
                    outputs[("disp", 0, scale)] = outputs[("disp", scale)]
            
            # unflip all outputs
            if do_flip:
                for k, v in outputs.items():
                    outputs[k] = torch.flip(v, dims=(-1, ))
        elif "unidepth" in cfg.model.name:
            if cfg.model.name in ["unidepth", 
                                  "unidepth_unprojector_vit", 
                                  "unidepth_unprojector_cnvnxtl"]:
                outputs = self.models["unidepth"](inputs)
            elif cfg.model.name in ["unidepth_extension_vit",
                                    "unidepth_extension_cnvnxtl"]:
                outputs = self.models["unidepth_extended"](inputs)

            input_f_id = 0
            outputs = {(key[0], input_f_id, key[1]): v for key, v in outputs.items()}

        input_f_id = 0
        scale = 0
        if not ("depth", input_f_id, scale) in outputs:
            disp = outputs[("disp", input_f_id, scale)]
            _, depth = disp_to_depth(disp, cfg.model.min_depth, cfg.model.max_depth)
            outputs[("depth", input_f_id, scale)] = depth

        self.compute_gauss_means(inputs, outputs)

        return outputs

    def target_tensor_image_dims(self, inputs):
        B, _, H, W = inputs["color", 0, 0].shape
        return B, H, W

    def compute_gauss_means(self, inputs, outputs):
        cfg = self.cfg
        input_f_id = 0
        scale = 0
        depth = outputs[("depth", input_f_id, scale)]
        B, _, H, W = depth.shape
        if ("inv_K_src", scale) in inputs:
            inv_K = inputs[("inv_K_src", scale)]
        else:
            inv_K = outputs[("inv_K_src", input_f_id, scale)]
        if self.cfg.model.gaussians_per_pixel > 1:
            inv_K = rearrange(inv_K[:,None,...].
                              repeat(1, self.cfg.model.gaussians_per_pixel, 1, 1),
                              'b n ... -> (b n) ...')
        xyz = self.backproject_depth[str(scale)](
            depth, inv_K
        )
        inputs[("inv_K_src", scale)] = inv_K
        if cfg.model.predict_offset:
            offset = outputs[("gauss_offset", input_f_id, scale)]
            if cfg.model.scaled_offset:
                offset = offset * depth.detach()
            offset = offset.view(B, 3, -1)
            zeros = torch.zeros(B, 1, H * W, device=depth.device)
            offset = torch.cat([offset, zeros], 1)
            xyz = xyz + offset # [B, 4, W*H]
        outputs[("gauss_means", input_f_id, scale)] = xyz

    def checkpoint_dir(self):
        return Path("checkpoints")
    
    def save_model(self, optimizer, step, ema=None):
        """Save model weights to disk
        """
        save_folder = self.checkpoint_dir()
        save_folder.mkdir(exist_ok=True, parents=True)

        save_path = save_folder / f"model_{step:07}.pth"
        logging.info(f"saving checkpoint to {str(save_path)}")

        model = ema.ema_model if ema is not None else self
        save_dict = {
            "model": model.state_dict(),
            "version": "1.0",
            "optimiser": optimizer.state_dict(),
            "step": step
        }
        torch.save(save_dict, save_path)

        num_ckpts = self.cfg.optimiser.num_keep_ckpts
        ckpts = sorted(list(save_folder.glob("model_*.pth")), reverse=True)
        if len(ckpts) > num_ckpts:
            for ckpt in ckpts[num_ckpts:]:
                ckpt.unlink()

    def load_model(self, weights_path, optimizer=None, device='cpu'):
        """Load model(s) from disk
        """
        weights_path = Path(weights_path)

        # determine if it is an old or new saving format
        if weights_path.is_dir() and weights_path.joinpath("encoder.pth").exists():
            self.load_model_old(weights_path, optimizer)
            return

        logging.info(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=torch.device(device))
        if "version" in state_dict and state_dict["version"] == "1.0":
            new_dict = {}
            for k, v in state_dict["model"].items():
                if "backproject_depth" in k:
                    new_dict[k] = self.state_dict()[k].clone()
                else:
                    new_dict[k] = v.clone()
            # for k, v in state_dict["model"].items():
            #     if "backproject_depth" in k and ("pix_coords" in k or "ones" in k):
            #         # model has these parameters set as a function of batch size
            #         # when batch size changes in eval this results in a loading error
            #         state_dict["model"][k] = v[:1, ...]
            self.load_state_dict(new_dict, strict=False)
        else:
            # TODO remove loading according to the old format
            for name in self.cfg.train.models_to_load:
                if name not in self.models:
                    continue
                self.models[name].load_state_dict(state_dict[name])

        # loading adam state
        if optimizer is not None:
            optimizer.load_state_dict(state_dict["optimiser"])
            self.step = state_dict["step"]

    def load_model_old(self, weights_folder, optimizer=None):
        for n in self.cfg.train.models_to_load:
            print(f"Loading {n} weights...")
            path = weights_folder / f"{n}.pth"
            if n not in self.models:
                continue
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = weights_folder / "adam.pth"
        if optimizer is not None and optimizer_load_path.is_file():
            print("Loading Adam weights")
            optimizer_state = torch.load(optimizer_load_path)
            optimizer.load_state_dict(optimizer_state["adam"])
            self.step = optimizer_state["step"]
