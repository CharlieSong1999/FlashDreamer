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
from torchvision.transforms.functional import to_pil_image
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
        # self.model_cfg_path = hf_hub_download(repo_id="einsafutdinov/flash3d", 
        #                                       filename="config_re10k_v1.yaml")
        # self.model_path = hf_hub_download(repo_id="einsafutdinov/flash3d", 
        #                                   filename="model_re10k_v2.pth")

        self.model_cfg_path = "./flash3d-hub/config_re10k_v1.yaml"
        self.model_path = "./flash3d-hub/model_re10k_v2.pth"

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


    def get_SE3_rotation_y(self, theta_degrees):
        """
        Return SE(3) matrix, rotation around y axix and no translation
        """

        theta = np.deg2rad(theta_degrees)
        R_y = np.array([
            [ np.cos(theta), 0, np.sin(theta)],
            [ 0,             1, 0            ],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        T = np.eye(4)
        T[:3, :3] = R_y

        return torch.tensor(T).cuda().float()

    def apply_mask_from_images(self, image_a, image_b):
        """
        Get mask from image_a and use it to filter image_b
        """
        image_a = image_a.convert('RGBA')
        image_b = image_b.convert('RGBA')

        if image_a.size != image_b.size:
            image_b = image_b.resize(image_a.size)

        image_a_np = np.array(image_a)
        image_b_np = np.array(image_b)

        rgb_channels = image_a_np[:, :, :3]
        rgb_nonzero = np.any(rgb_channels != 0, axis=2)
        mask = rgb_nonzero

        mask_3d = np.stack([mask]*4, axis=-1)
        output_np = np.zeros_like(image_b_np)
        output_np[mask_3d] = image_b_np[mask_3d]
        output_tensor = torch.from_numpy(output_np).permute(2, 0, 1).float() / 255.0

        return output_tensor, mask

    def check_input_image(self, input_image):
        if input_image is None:
            raise FileNotFoundError("Input image not found!")

    def preprocess(self, image, dynamic_size=False, padding=True):
        """图片预处理"""
        h, w = image.size
        size = 32
        if dynamic_size:
            while max(h, w) // size > 20: # initially 20
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

        result = self.model(inputs) # inference results of current frame
        outputs = postprocess(result,
                            num_gauss=2,
                            h=self.cfg.dataset.height,
                            w=self.cfg.dataset.width,
                            pad=self.cfg.dataset.pad_border_aug)
        
        outputs_1_gauss = postprocess(result,
                            num_gauss=1,
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
        # w2c = torch.tensor([
        #                     [1.0, 0.0, 0.0, 0.0],  
        #                     [0.0, 1.0, 0.0, 0.0], 
        #                     [0.0, 0.0, 1.0, 0.1], 
        #                     [0.0, 0.0, 0.0, 1.0]
        #                 ], dtype=torch.float32)

        w2c = self.get_SE3_rotation_y(20)
    
        im, radius = self.renderer.render(outputs, w2c)
        self.renderer.save_image(im, root_directory+"render_test.png")

        # render 1 gauss per pixel
        im_1_gauss, radius = self.renderer.render(outputs_1_gauss, w2c)
        self.renderer.save_image(im_1_gauss, root_directory+"render_test_1.png")

        image_a_pil = to_pil_image(im_1_gauss)
        image_b_pil = to_pil_image(im)

        masked_img, mask = self.apply_mask_from_images(image_a_pil, image_b_pil)
        self.renderer.save_image(masked_img, root_directory+"masked_rendered.png")

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
