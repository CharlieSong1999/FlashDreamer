import sys
import os
import shutil

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
work_directory = os.path.join(current_directory, 'flash3d')
sys.path.append(work_directory)
sys.path.append(parent_directory)

from omegaconf import OmegaConf
from torchvision.transforms.functional import resize
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
from vlm_diffusion_pipeline import main as generate_diffusion_img
from torchvision import transforms
from img2video import create_video_from_images


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

        # self.model_cfg_path = "./flash3d-hub/config_re10k_v1.yaml"
        # self.model_path = "./flash3d-hub/model_re10k_v2.pth"

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

        self.imgs = []  # 暂时手动存放diffusion图片
        self.diffusion_img = None
        self.w2c = []  # pre-defined w2c
        self.index = 0
        self.mask = None
        self.gt_img = []  # 输入图片+生成的图片

        self.optimize_num_iters = 100

    def get_SE3_rotation_y(self, theta_degrees):
        """
        Return SE(3) matrix, rotation around y axix and no translation
        """

        theta = np.deg2rad(theta_degrees)
        R_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
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

        mask_3d = np.stack([mask] * 4, axis=-1)
        output_np = np.zeros_like(image_b_np)
        output_np[mask_3d] = image_b_np[mask_3d]
        output_tensor = torch.from_numpy(output_np).permute(2, 0, 1).float() / 255.0

        return output_tensor, mask

    def check_input_image(self, input_image):
        """检查图片是否存在"""
        if input_image is None:
            raise FileNotFoundError("Input image not found!")
        return resize(input_image, [680, 1200])

    def preprocess(self, image, dynamic_size=False, padding=True):
        """图片预处理"""
        h, w = image.size
        size = 32
        if dynamic_size:
            while max(h, w) // size > 20:  # initially 20
                size *= 2
            crop_image = TTF.center_crop(image, (w // size * size, h // size * size))
            resize_image = TTF.resize(crop_image, (w // size * 32, h // size * 32),
                                      interpolation=TT.InterpolationMode.BICUBIC)
            self.cfg.dataset.width, self.cfg.dataset.height = resize_image.size
        else:
            self.cfg.dataset.height, self.cfg.dataset.width = 160, 288
            resize_image = TTF.resize(
                image, (self.cfg.dataset.height, self.cfg.dataset.width),
                interpolation=TT.InterpolationMode.BICUBIC
            )
        if padding:
            resize_image.save(current_directory + f"/imgs/{self.index}_resized.png")
            input_image = self.pad_border_fn(resize_image)
            self.cfg.dataset.pad_border_aug = 32
        else:
            input_image = resize_image
            self.cfg.dataset.pad_border_aug = 0
        self.model.set_backproject()
        return input_image

    def apply_transformation(self, point_cloud, T):
        """点云坐标变换"""
        ones = torch.ones((point_cloud.shape[0], 1), device=point_cloud.device)
        homogeneous_points = torch.cat((point_cloud, ones), dim=1)

        transformed_points = torch.mm(homogeneous_points, T.t())
        return transformed_points[:, :3]

    def optimize_map(self, params):
        params_group = []

        for key in params:
            if isinstance(params[key], torch.Tensor):
                params[key] = params[key].detach().to(self.device)
                params[key].requires_grad_(True)

        params_group = [{'params': [params[key]]} for key in params]
        optimizer = torch.optim.Adam(params_group, lr=1e-3)
        num_iters = self.optimize_num_iters
        for i in range(num_iters):
            optimizer.zero_grad()
            loss = 0.0
            num_gt = len(self.gt_img)
            # 为每个gt图片赋予不同的权重, 越靠近当前视角的权重越大
            # weights = torch.tensor([0.1 ** (i) for i in range(num_gt)], dtype=torch.float32, device=self.device).__reversed__()
            for k in range(num_gt):
                # 渲染
                rendered, radius = self.renderer.render(params, self.w2c[k])
                rendered = rendered[:, 32:352, 32:608]

                # 保存渲染结果
                if i == 0:
                    self.renderer.save_image(rendered, current_directory + f"/imgs/{self.index}_{k}_start_render.png")
                if i == num_iters - 1:
                    self.renderer.save_image(rendered, current_directory + f"/imgs/{self.index}_{k}_end_render.png")

                # 计算损失
                gt = self.gt_img[k].squeeze(0)[:, 32:352, 32:608]
                loss += torch.abs(rendered - gt).sum()  # L1 loss + 权重
                # loss += torch.abs(rendered - gt).sum() * weights[k] # L1 loss + 权重

            # 反向传播
            loss /= num_gt
            loss.backward()

            # 更新参数
            optimizer.step()

            print(f"Tracking Iteration {i + 1}, Loss: {loss.item():.4f}")
        return params

    @spaces.GPU()
    def reconstruct_and_export(self, image, output_dir, image_path, num_gauss=2):
        if (self.index == 0):
            image = self.to_tensor(image).to(self.device).unsqueeze(0)

        else:
            image = self.diffusion_img

        save_image(image, current_directory + f"/imgs/{self.index}_inputpre.png")
        self.gt_img.append(image)

        inputs = {
            ("color_aug", 0, 0): image,
        }

        # flash3d输出
        result = self.model(inputs)

        # 2层3dg
        outputs = postprocess(result,
                              num_gauss=2,
                              h=self.cfg.dataset.height,
                              w=self.cfg.dataset.width,
                              pad=self.cfg.dataset.pad_border_aug)
        # 1层3dg
        outputs_1_gauss = postprocess(result,
                                      num_gauss=1,
                                      h=self.cfg.dataset.height,
                                      w=self.cfg.dataset.width,
                                      pad=self.cfg.dataset.pad_border_aug)

        # 处理初始输入图片，添加到地图中
        if (self.index == 0):
            self.map_param_2 = outputs
            self.map_param_2['rotations'] = torch.tensor(self.map_param_2['rotations']).to('cuda')

            self.map_param_1 = outputs_1_gauss
            self.map_param_1['rotations'] = torch.tensor(self.map_param_1['rotations']).to('cuda')

            self.map_param_2 = self.optimize_map(self.map_param_2)

        # 处理生成的图片，按照mask添加新元素，变换到世界坐标下，添加到地图中
        if (self.index != 0):
            w2c = self.w2c[self.index]

            self.cur_param_2 = outputs
            self.cur_param_2['rotations'] = torch.tensor(self.cur_param_2['rotations']).to('cuda')

            self.cur_param_1 = outputs_1_gauss
            self.cur_param_1['rotations'] = torch.tensor(self.cur_param_1['rotations']).to('cuda')

            c2w = torch.inverse(w2c)
            self.cur_param_1['means'] = self.apply_transformation(self.cur_param_1['means'], c2w.to('cuda'))
            self.cur_param_2['means'] = self.apply_transformation(self.cur_param_2['means'], c2w.to('cuda'))

            # 保留新增的部分
            mask = ~torch.tensor(self.mask).view(-1)
            mask_2 = mask.repeat(2)
            # update global map
            for key in self.map_param_1.keys():

                original_tensor = self.map_param_1[key].to('cuda')
                updated_tensor = torch.tensor(self.cur_param_1[key]).to('cuda')
                updated_tensor = updated_tensor[mask]

                if isinstance(updated_tensor, np.ndarray):
                    updated_tensor = torch.tensor(updated_tensor).to('cuda')

                self.map_param_1[key] = torch.cat((original_tensor, updated_tensor), dim=0)

                original_tensor = self.map_param_2[key].to('cuda')
                updated_tensor = torch.tensor(self.cur_param_2[key]).to('cuda')
                updated_tensor = updated_tensor[mask_2]

                if isinstance(updated_tensor, np.ndarray):
                    updated_tensor = torch.tensor(updated_tensor).to('cuda')

                self.map_param_2[key] = torch.cat((original_tensor, updated_tensor), dim=0)

            self.map_param_2 = self.optimize_map(self.map_param_2)

        if ((self.index + 1) < len(self.w2c)):
            # 新视角下渲染
            w2c = self.w2c[self.index + 1]
            im_original, radius = self.renderer.render(self.map_param_2, w2c)
            im = im_original[:, 32:352, 32:608]
            self.renderer.save_image(im, current_directory + f"/imgs/{self.index}_render_2gauss.png")

            # render 1 gauss per pixel
            im_1_gauss_original, radius = self.renderer.render(self.map_param_1, w2c)
            im_1_gauss = im_1_gauss_original[:, 32:352, 32:608]
            self.renderer.save_image(im_1_gauss, current_directory + f"/imgs/{self.index}_render_1gauss.png")

            image_a_pil = to_pil_image(im_1_gauss)
            image_b_pil = to_pil_image(im)
            masked_img, mask = self.apply_mask_from_images(image_a_pil, image_b_pil)

            self.mask = mask  # mask for the diffusion and adding new 3dg
            mask_render_path = current_directory + f"/imgs/{self.index}_masked_rendered.png"
            self.renderer.save_image(masked_img, mask_render_path)

            # 获取diffusion的mask
            image_a_pil = to_pil_image(im_1_gauss_original)
            image_b_pil = to_pil_image(im_original)
            masked_img_diffusion, mask_diffusion = self.apply_mask_from_images(image_a_pil, image_b_pil)
            mask_diffusion = ~torch.tensor(mask_diffusion)
            mask_diffusion = mask_diffusion.to(torch.float32)

            # input of diffusion
            mask_render_path_diffusion = current_directory + f"/imgs/{self.index}_masked_rendered_original.png"
            self.renderer.save_image(masked_img_diffusion, mask_render_path_diffusion)
            mask_path_diffusion = current_directory + f"/imgs/{self.index}_mask_diffusion.png"
            self.renderer.save_image(mask_diffusion, mask_path_diffusion)

            self.model.to('cpu')

            # 生成图片
            diffusion_img = generate_diffusion_img(original_input_img_path=image_path,
                                                   image_path=mask_render_path_diffusion, mask_path=mask_path_diffusion,
                                                   prompt_question='What is the scene of the image? Answer in a sentence.',
                                                   prompt_diffusion=None,
                                                   base_model='stable-diffusion-v2', index=self.index)  # 512*512

            self.model.to(self.device)

            transform = transforms.Compose([
                transforms.Resize((384, 640)),  # 先调整大小为 (height, width)
                transforms.CenterCrop((320, 576)),  # 再中心裁剪为 (height, width)
                transforms.Pad(padding=32, fill=(0, 0, 0))  # 最后添加 32 像素的填充，填充颜色为黑色
            ])
            diffusion_img = transform(diffusion_img)

            self.diffusion_img = self.to_tensor(diffusion_img).to(self.device).unsqueeze(0)  # [1, 3, 384, 640]

        self.index += 1

    def save_ply(self, output_dir):
        # 保存ply文件
        print(f'Save the ply files to directory:{output_dir}')
        save_ply(self.map_param_2,
                 path=os.path.join(output_dir, 'demo.ply'))
        save_ply(self.map_param_1,
                 path=os.path.join(output_dir, 'demo_1.ply'))

    def run(self, img_path, output_dir=None, dynamic_size=True, padding=True):
        img = Image.open(img_path).convert("RGB")
        img = self.check_input_image(img)  # add resize to [1200, 680]
        img = self.preprocess(img, dynamic_size=dynamic_size, padding=padding)

        self.reconstruct_and_export(img, output_dir, img_path)


def clear_cache(folder_path):
    """
    Clears all contents within the specified folder.

    Parameters:
    folder_path (str): The path to the folder to be cleared.
    """
    # Ensure the folder exists
    if os.path.exists(folder_path):
        # Iterate over all contents in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # Remove directories and files
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the entire directory
            else:
                os.remove(file_path)  # Remove individual file

        print(f"All contents in '{folder_path}' have been cleared.")
    else:
        print(f"The folder '{folder_path}' does not exist.")

if __name__ == "__main__":

    # clear cache
    clear_cache('./imgs')
    clear_cache('./output')

    '''
    # img_path = os.path.join(current_directory, './input/subset/frame001747.jpg')
    # img_path = os.path.join(current_directory, './input/selected/8a_gameroom2.jpg')
    img_path = os.path.join(current_directory, './input/0_4.jpg')
    output_path = os.path.join(current_directory, 'demo')

    reconstructor = Flash3DReconstructor()

    # 初始视角(input image的视角)
    w2c_0 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    w2c_back = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.2],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    # 添加视角，w2c_0为初始视角
    reconstructor.w2c.append(w2c_0)

    angles = [-10, 10]
    for angle in angles:
        # reconstructor.w2c.append(w2c_back)
        w2c = reconstructor.get_SE3_rotation_y(angle)
        reconstructor.w2c.append(w2c)

    for i in range(len(reconstructor.w2c)):
        print('Processing image', i)
        reconstructor.run(img_path=img_path,
                          output_dir=output_path)

    # Optimize
    reconstructor.map_param_1 = reconstructor.optimize_map(reconstructor.map_param_1)

    # Render
    angles = [-10, 10]
    for i, angle in enumerate(angles):
        temp_w2c = reconstructor.get_SE3_rotation_y(angle)

        im, radius = reconstructor.renderer.render(reconstructor.map_param_1, temp_w2c)
        im = im[:, 32:352, 32:608]
        reconstructor.renderer.save_image(im, current_directory + f'/output/{angle}_render.png')
    '''

    # Process each image in the folder
    input_folder = './input'
    output_folder = './output'
    angles_list = [[-20, 20], [-10, 10], [-30, 30]]
    #angles = [-10,10]#[-20, 20],[-10,10],[-30,30]

    w2c_0 = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    for angles in angles_list:
        for img_name in os.listdir(input_folder):
            clear_cache('./imgs') # Start with each image without
            img_path = os.path.join(input_folder, img_name)
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Skip non-image files
                continue

            reconstructor = Flash3DReconstructor()
            reconstructor.w2c.append(w2c_0)

            # Render each image at specified angles
            for angle in angles:
                w2c = reconstructor.get_SE3_rotation_y(angle)
                reconstructor.w2c.append(w2c)
                angle_folder = os.path.join(output_folder, f"{angle}")
                os.makedirs(angle_folder, exist_ok=True)

            for i in range(len(reconstructor.w2c)):
                print('Processing image', i)
                reconstructor.run(img_path=img_path)

            reconstructor.map_param_1 = reconstructor.optimize_map(reconstructor.map_param_1)

            for i, angle in enumerate(angles):
                temp_w2c = reconstructor.get_SE3_rotation_y(angle)

                im, radius = reconstructor.renderer.render(reconstructor.map_param_1, temp_w2c)
                im = im[:, 32:352, 32:608]

                output_img_path = os.path.join(output_folder, f"{angle}",
                                               f"{os.path.splitext(img_name)[0]}_{angle}.png")
                reconstructor.renderer.save_image(im, output_img_path)

                print(f"Processed image '{img_name}' at angle {angle}, saved to '{output_img_path}'")
            # break
            del(reconstructor)
    print("All images processed with specified angles.")

    # current_directory = os.path.dirname(os.path.abspath(__file__))
    # image_folder = current_directory + '/rotate_demo'
    # video_name = current_directory + '/rotate_demo/output_video.mp4'
    # create_video_from_images(image_folder, video_name)
    #
    # reconstructor.save_ply(image_folder)
