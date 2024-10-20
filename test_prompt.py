	
import sys
sys.path.append('./flash3d')
sys.path.append('./flash3d/flash3d')
from flash3d.generator_test import Flash3DReconstructor
import torch
import types
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import to_pil_image
from flash3d.flash3d.util.export_param import postprocess
import numpy as np
from torchvision import transforms
from vlm_diffusion_pipeline import main as generate_diffusion_img
from matplotlib import pyplot as plt
from diffusers.utils import load_image, make_image_grid
import pdb


# Define the decorator to add the function to the instance of the reconstructor
def decorator_add_function_to_instance(instance, func):
    setattr(instance, func.__name__, types.MethodType(func, instance))

flash3dreconstructor = Flash3DReconstructor()
setattr(flash3dreconstructor, 'add_function_to_instance', types.MethodType(decorator_add_function_to_instance, flash3dreconstructor))



# Preprocess the input image, get the flash3d output, then the postprocess
@flash3dreconstructor.add_function_to_instance
def flash3d_postprocess(self, index, image):
    if (index == 0):
        image = self.to_tensor(image).to(self.device).unsqueeze(0)

    else:
        image = self.diffusion_img

    save_image(image, current_directory+f"/imgs/{index}_inputpre.png")
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
    
    return outputs, outputs_1_gauss

# 处理初始输入图片，添加到地图中
# Directly copy from Jiaqi's code
@flash3dreconstructor.add_function_to_instance
def flash3d_initial_map(self, outputs, outputs_1_gauss):
    self.map_param_2 = outputs
    self.map_param_2['rotations'] = torch.tensor(self.map_param_2['rotations']).to('cuda')

    self.map_param_1 = outputs_1_gauss
    self.map_param_1['rotations'] = torch.tensor(self.map_param_1['rotations']).to('cuda')

    self.map_param_2 = self.optimize_map(self.map_param_2)

# 处理生成的图片，按照mask添加新元素，变换到世界坐标下，添加到地图中
# Directly copy from Jiaqi's code
@flash3dreconstructor.add_function_to_instance
def flash3d_additional_map(self, index, outputs, outputs_1_gauss):
    w2c = self.w2c[index]

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

@flash3dreconstructor.add_function_to_instance
def flash3d_prepare_img_mask_for_diffusion(self, index):
    # 新视角下渲染
    w2c = self.w2c[index+1]
    im_original, radius = self.renderer.render(self.map_param_2, w2c)
    im = im_original[:, 32:352, 32:608]
    self.renderer.save_image(im, current_directory+f"/imgs/{index}_render_2gauss.png")

    # render 1 gauss per pixel
    im_1_gauss_original, radius = self.renderer.render(self.map_param_1, w2c)
    im_1_gauss = im_1_gauss_original[:, 32:352, 32:608]
    self.renderer.save_image(im_1_gauss, current_directory+f"/imgs/{index}_render_1gauss.png")

    image_a_pil = to_pil_image(im_1_gauss)
    image_b_pil = to_pil_image(im)
    masked_img, mask = self.apply_mask_from_images(image_a_pil, image_b_pil)

    self.mask = mask # mask for the diffusion and adding new 3dg
    mask_render_path = current_directory+f"/imgs/{index}_masked_rendered.png"
    self.renderer.save_image(masked_img, mask_render_path)

    # 获取diffusion的mask
    image_a_pil = to_pil_image(im_1_gauss_original)
    image_b_pil = to_pil_image(im_original)
    masked_img_diffusion, mask_diffusion = self.apply_mask_from_images(image_a_pil, image_b_pil)
    mask_diffusion = ~torch.tensor(mask_diffusion)
    mask_diffusion = mask_diffusion.to(torch.float32)


    # input of diffusion
    mask_render_path_diffusion = current_directory+f"/imgs/{index}_masked_rendered_original.png"
    self.renderer.save_image(masked_img_diffusion, mask_render_path_diffusion)
    mask_path_diffusion = current_directory+f"/imgs/{index}_mask_diffusion.png"
    self.renderer.save_image(mask_diffusion, mask_path_diffusion)

    return mask_render_path_diffusion, mask_path_diffusion

@flash3dreconstructor.add_function_to_instance
def flash3d_post_process_diffusion_img(self, diffusion_img):
    transform = transforms.Compose([
                transforms.Resize((384, 640)),  # 先调整大小为 (height, width)
                transforms.CenterCrop((320, 576)),  # 再中心裁剪为 (height, width)
                transforms.Pad(padding=32, fill=(0, 0, 0))  # 最后添加 32 像素的填充，填充颜色为黑色
            ])
    diffusion_img = transform(diffusion_img)

    self.diffusion_img = self.to_tensor(diffusion_img).to(self.device).unsqueeze(0) # [1, 3, 384, 640]

@flash3dreconstructor.add_function_to_instance
def flash3d_final_process(self, ):
    reconstructor = self
    # 优化1 layer的map
    reconstructor.map_param_1 = reconstructor.optimize_map(reconstructor.map_param_1)

    # 不同视角渲染地图并保存
    for i in range(15, -1, -1):
        temp_w2c = reconstructor.get_SE3_rotation_y(i)

        im, radius = reconstructor.renderer.render(reconstructor.map_param_1, temp_w2c)
        im = im[:, 32:352, 32:608]
        reconstructor.renderer.save_image(im, current_directory+f'/rotate_demo/{15-i}_render.png')

    for i in range(0, 30, 1):
        temp_w2c = reconstructor.get_SE3_rotation_y(i)

        im, radius = reconstructor.renderer.render(reconstructor.map_param_1, temp_w2c)
        im = im[:, 32:352, 32:608]
        reconstructor.renderer.save_image(im, current_directory+f'/rotate_demo/{i+16}_render.png')



    # Set the path to the input image and the output directory, the input image is the very first image of the whole pipeline
intial_img_path = './flash3d/frame000652.jpg'
output_path = './flash3d-output'
current_directory = './flash3d-cache'

# 初始视角(input image的视角)
w2c_0 = torch.tensor([
                    [1.0, 0.0, 0.0, 0.0],  
                    [0.0, 1.0, 0.0, 0.0], 
                    [0.0, 0.0, 1.0, 0.0], 
                    [0.0, 0.0, 0.0, 1.0]
                ], dtype=torch.float32)

# w2c back denotes the transformation matrix for the camera to backward while maintaining the same view angle
# 0.2 is roungly the distance (not entirely sure) between the camera and the object, adjust this value to adjust the distance between the camera and the object
# If want to combine it with rotation, just multiply the rotation matrix with this matrix
backward_distance = 0.2
w2c_back = torch.tensor([
                    [1.0, 0.0, 0.0, 0.0],  
                    [0.0, 1.0, 0.0, 0.0], 
                    [0.0, 0.0, 1.0, backward_distance], 
                    [0.0, 0.0, 0.0, 1.0]
                ], dtype=torch.float32)

# 添加视角，w2c_0为初始视角
flash3dreconstructor.w2c.append(w2c_0)

# Prepare the input image, which refers to the very first image of the whole pipeline
img = Image.open(intial_img_path).convert("RGB")
if img is not None:
    print("Image loaded successfully.")
else:
    print("Failed to load the image.")

# Add the transformation matrix you want to apply to the camera
rotate_angle = [5, 10, 15, 20, 30]
for angle in rotate_angle:
    rotate_matrix = flash3dreconstructor.get_SE3_rotation_y(angle)
    flash3dreconstructor.w2c.append(rotate_matrix) # This line to add the rotation matrix to the camera
    # rotate_matrix = flash3dreconstructor.get_SE3_rotation_y(rotate_angle)
    # flash3dreconstructor.w2c.append(rotate_matrix) # This line to add the rotation matrix to the camera


flash3dreconstructor.check_input_image(img)
img = flash3dreconstructor.preprocess(img, dynamic_size=True, padding=True)

    # Define the current loop index, the loop includes the entire pipeline, from the input image to the final output, it normally depends on the number of transformations
    # you add to the w2c list of the reconstructor
    # Note the input image of the first loop (current_loop_index = 0) is the image you prepared above
    # The output image of the first loop is the input image of the second loop, and so on
current_loop_index = 5


flash3dreconstructor.model.to('cuda')

for i in range(current_loop_index):
    outputs, outputs_1_gauss = flash3dreconstructor.flash3d_postprocess(current_loop_index, img)
    if current_loop_index == 0:
        flash3dreconstructor.flash3d_initial_map(outputs, outputs_1_gauss)
    else:
        flash3dreconstructor.flash3d_additional_map(current_loop_index, outputs, outputs_1_gauss)

    if current_loop_index + 1 < len(flash3dreconstructor.w2c):
        mask_render_path_diffusion, mask_path_diffusion = flash3dreconstructor.flash3d_prepare_img_mask_for_diffusion(current_loop_index)


    # Display the render image and mask image
    if current_loop_index + 1 < len(flash3dreconstructor.w2c):
        rendered_img = Image.open(mask_render_path_diffusion)
        mask_img = Image.open(mask_path_diffusion)

        grid_img = make_image_grid([rendered_img, mask_img], rows=1, cols=2)
        plt.imshow(grid_img)
        plt.axis('off')
        plt.savefig(f'./imgs/before_inpainting_{current_loop_index}.jpg')

    Prompt_diffusion = 'A indoor scene, a room, a window, two sofas.'

    if current_loop_index + 1 < len(flash3dreconstructor.w2c):
        diffusion_img = generate_diffusion_img(image_path=mask_render_path_diffusion, mask_path=mask_path_diffusion,
                                                        prompt_question=None,
                                                        prompt_diffusion=Prompt_diffusion,
                                                        base_model='stable-diffusion-xl', index=current_loop_index, strength=1.0,
                                                        negative_prompt="bad architecture, inconsistent, poor details, blurry") # 512*512

    # Display the diffusion image
    if current_loop_index + 1 < len(flash3dreconstructor.w2c):
        plt.imshow(diffusion_img)
        plt.axis('off')
        plt.savefig(f'./imgs/diffusion_{current_loop_index}.jpg')

    # Postprocess the diffusion image
    flash3dreconstructor.flash3d_post_process_diffusion_img(diffusion_img)

    flash3dreconstructor.flash3d_final_process()