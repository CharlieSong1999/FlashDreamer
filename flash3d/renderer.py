import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision.utils import save_image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

class Flash3DRenderer:
    def __init__(self):
        pass

    def flash3d2rendervar(self, params):

        means = torch.tensor(params['means'], device="cuda").float() if isinstance(params['means'], np.ndarray) else params['means'].float()
        harmonics = torch.tensor(params['harmonics'], device="cuda").float() if isinstance(params['harmonics'], np.ndarray) else params['harmonics'].float()
        rotations = torch.tensor(params['rotations'], device="cuda").float() if isinstance(params['rotations'], np.ndarray) else params['rotations'].float()
        opacities = torch.tensor(params['opacities'], device="cuda").float() if isinstance(params['opacities'], np.ndarray) else params['opacities'].float()
        scales = torch.tensor(params['scales'], device="cuda").float() if isinstance(params['scales'], np.ndarray) else params['scales'].float()

        # Check if Gaussians are Isotropic
        if scales.shape[1] == 1:
            scales = torch.tile(scales, (1, 3))

        # Initialize Render Variables
        rendervar = {
            'means3D': means,
            # NOTE
            # 理论上renderer只需要将sh参数输入进去，因为渲染出现问题且尚未解决，于是这里手动计算了rgb，
            # 使得渲染可以进行，这也意味着每个3d Gaussian的颜色变成了各向同性 
            'colors_precomp': torch.clamp(0.28209479177387814 * harmonics + 0.5, 0, 1),
            'rotations': rotations,
            'opacities': opacities,
            'scales': scales,
            'means2D': torch.zeros_like(means, requires_grad=True, device="cuda") + 0
        }

        return rendervar

    def setup_camera(self, w2c, near=0.01, far=20):

        # Replica数据集的内参
        k = torch.tensor([
            [600.0, 0, 599.5],
            [0, 600.0, 339.5],
            [0, 0, 1]
        ])

        # 预处理后输入flash3d的图片大小，也是渲染时图片的大小
        w = 640
        h = 384
        # w = 576
        # h = 320
        fx, fy, cx, cy = k[0][0] / 2, k[1][1] / 2, k[0][2] / 2, k[1][2] / 2

        # 设置观测视角的外参
        w2c = torch.tensor(w2c).cuda().float()
        cam_center = torch.inverse(w2c)[:3, 3]
        w2c = w2c.unsqueeze(0).transpose(1, 2)
        opengl_proj = torch.tensor([
            [2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
            [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
            [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
            [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)

        full_proj = w2c.bmm(opengl_proj)
        cam = Camera(
            image_height=h,
            image_width=w,
            tanfovx=w / (2 * fx),
            tanfovy=h / (2 * fy),
            bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj,
            sh_degree=0,
            campos=cam_center,
            prefiltered=False
        )
        return cam

    def render(self, params, w2c):
        # Initialize Render Variables
        rendervar = self.flash3d2rendervar(params)

        # RGB Rendering
        rendervar['means2D'].retain_grad()

        cam = self.setup_camera(w2c)
        im, radius, _, = Renderer(raster_settings=cam)(**rendervar)
        return im, radius

    def save_image(self, image_tensor, filename):
        # Ensure the image tensor is in the correct range [0, 1]
        image_tensor = image_tensor.clamp(0, 1)
        
        # Use torchvision's save_image function to save the tensor as an image
        save_image(image_tensor, filename)
        print(f"Image saved as {filename}")
