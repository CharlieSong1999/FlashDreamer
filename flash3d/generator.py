import sys
import os
root_directory = os.path.dirname(os.path.abspath(__file__))
work_directory = os.path.join(root_directory, 'flash3d')
sys.path.append(work_directory)

from omegaconf import OmegaConf
import spaces
import torch
import torchvision.transforms as TT
import torchvision.transforms.functional as TTF
from huggingface_hub import hf_hub_download
from PIL import Image
import json
from networks.gaussian_predictor import GaussianPredictor
from util.export_param import postprocess, save_ply

import numpy as np
from renderer import Flash3DRenderer
from torchvision.utils import save_image

class Flash3DReconstructor:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        # 加载模型
        self.model_cfg_path = hf_hub_download(repo_id="einsafutdinov/flash3d", 
                                              filename="config_re10k_v1.yaml")
        self.model_path = hf_hub_download(repo_id="einsafutdinov/flash3d", 
                                          filename="model_re10k_v2.pth")

        self.cfg = OmegaConf.load(self.model_cfg_path)
        self.model = GaussianPredictor(self.cfg)
        self.device = torch.device(self.device)
        self.model.load_model(self.model_path)
        self.model.to(self.device)

        self.pad_border_fn = TT.Pad((self.cfg.dataset.pad_border_aug, self.cfg.dataset.pad_border_aug))
        self.to_tensor = TT.ToTensor()

        self.map_param = None
        # 加载渲染器
        self.renderer = Flash3DRenderer()


    def check_input_image(self, input_image):
        if input_image is None:
            raise FileNotFoundError("Input image not found!")

    def preprocess(self, image, dynamic_size=False, padding=True):
        """图片预处理"""
        h, w = image.size
        size = 32
        if dynamic_size:
            while max(h, w) // size > 10: # initially 20
                size *= 2
            crop_image = TTF.center_crop(image, (w // size * size, h // size * size))
            resize_image = TTF.resize(crop_image, (w // size * 32, h // size * 32), interpolation=TT.InterpolationMode.BICUBIC)
            self.cfg.dataset.width, self.cfg.dataset.height = resize_image.size
        else:
            self.cfg.dataset.height, self.cfg.dataset.width = 160, 288
            resize_image = TTF.resize(
                image, (self.cfg.dataset.height, self.cfg.dataset.width), 
                interpolation=TT.InterpolationMode.BICUBIC
            )
        if padding:
            resize_image.save(root_directory+"resized.png")
            input_image = self.pad_border_fn(resize_image)
            self.cfg.dataset.pad_border_aug = 32
        else:
            input_image = resize_image
            self.cfg.dataset.pad_border_aug = 0
        self.model.set_backproject()
        return input_image

    @spaces.GPU()
    def reconstruct_and_export(self, image, output_dir, num_gauss=2):
        image = self.to_tensor(image).to(self.device).unsqueeze(0)
        save_image(image, root_directory+"inputpre.png")
        inputs = {
            ("color_aug", 0, 0): image,
        }

        outputs = self.model(inputs) # inference results of current frame
        outputs = postprocess(outputs,
                            num_gauss=num_gauss,
                            h=self.cfg.dataset.height,
                            w=self.cfg.dataset.width,
                            pad=self.cfg.dataset.pad_border_aug)
        # 初始视角
        # w2c = torch.tensor([
        #                     [1.0, 0.0, 0.0, 0.0],  
        #                     [0.0, 1.0, 0.0, 0.0], 
        #                     [0.0, 0.0, 1.0, 0.0], 
        #                     [0.0, 0.0, 0.0, 1.0]
        #                 ], dtype=torch.float32)

        # 偏移后的视角
        w2c = torch.tensor([[ 0.99569565,  0.00377457, -0.09260627, -0.51011687],
                            [-0.02098674,  0.98240015, -0.18560577,  0.0884598 ],
                            [ 0.09027583,  0.18675037,  0.97825077, -0.20966021],
                            [ 0.        ,  0.        ,  0.        ,  1.        ]]).cuda().float()
        
        im, radius = self.renderer.render(outputs, w2c)
        self.renderer.save_image(im, root_directory+"render_test.png")
        self.map_param = outputs
        outputs['rotations'] = torch.tensor(outputs['rotations']).clone().detach()

        save_ply(self.map_param, 
                path=os.path.join(output_dir, 'demo.ply'))

    def run(self, img_path, output_dir, dynamic_size=True, padding=True):
        img = Image.open(img_path).convert("RGB")
        self.check_input_image(img)
        img = self.preprocess(img, dynamic_size=dynamic_size, padding=padding)
        self.reconstruct_and_export(img, output_dir)

if __name__ == "__main__":
    reconstructor = Flash3DReconstructor()

    img_path = os.path.join(root_directory, 'frame000652.jpg')
    output_path = os.path.join(root_directory, 'demo')

    reconstructor.run(img_path= img_path, 
                      output_dir=output_path)
