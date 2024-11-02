# FlashDreamer

<img src="https://gitee.com/zhu-liyun2000/typora_imgs/raw/master/img/202411022105129.png" alt="image-20241102210536880" style="zoom: 50%;" />

This is a implementation of our project **"Enhancing Monocular 3D Scene Completion with Diffusion Model"** (ANU COMP 8536 project). We provide the implementation for reconstructing a complete 3D scene from a single image, significantly reducing the need for multi-view inputs.

Our approach leverages a vision-language model to generate descriptive prompts for the scene, guiding a diffusion model to produce images from various perspectives, which are then fused to form a cohesive 3D reconstruction.



## Installation

Please refer to **requirements.txt** for detailed environment setup requirements.

For further information on the Vision Language Model (VLM) and Diffusion model used in this project, please visit the websites for [llama-3.1-8B-vision-378](https://huggingface.co/qresearch/llama-3.1-8B-vision-378) and [Stable-Diffusion-v2](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting).



## Quick Start

```python
python3 test_prompt.py --output_path './output_imgs' \
        --intial_img_path './flash3d/frame000652.jpg' \
        --prompt_question 'Please describe the scene shortly.' \
        --rotate_angle_list '-30, -20, -10, 0, 10, 20, 30' \
        --base_model 'stable-diffusion-xl' \
        --optimize_num_iters 500 
```



## Acknowledgement

This codebase is built on top of [Flash3D](https://github.com/eldar/flash3d), and we thank the authors for their work.



## Development Updates

### First commit

The current pipeline consists of (1) generating text prompts using Llama 3.1 and (2) completing images based on a given image, a mask image, and the prompt.

You can bypass the Llama-generated prompts by setting the --prompt_diffusion input parameter.

> Example prompt in the screenshot: The image is a photo taken from a first-person perspective, showing a person walking into a room. The room appears to be a modern office space with a large window and a white couch.

TODO(issues):

1. The prompts generated directly by Llama 3.1 are not very satisfactory; you can try adjusting them yourself.
2. Mask images still need to be generated manually; the current pipeline only supports directly reading mask files (using the --mask_path input parameter).
3. Currently, the Hugging Face inpaint pipeline seems to only support 512x512 images; inputting other sizes may result in incorrect output image sizes. Therefore, images need to be manually resized; the images in the imgs folder have already been resized.

### 20240930 update

- Added support for Llama 3.2; you need to specify the --vlm_model field.
- Added Stable Diffusion XL and Stable Diffusion V2; you need to specify the --base_model field.
- `pre-process-img.py` is used to preprocess images rendered by Flash3D into a format acceptable for diffusion: 512x512 (1024x1024 for Stable Diffusion XL). It will cover pixel value areas with 0 and non-pixel areas with 255 based on the flash3dmasked file. Both the mask and flash3drender_test images will be resized.
- `flash3d_mask_resize_diffuse.bat` is a Windows batch file that automates the entire pipeline (though on my machine, the last line always fails to find a file, likely due to a Windows path issue) as a reference.


### 20241020 update

- developed by liyun
- Added test_prompt.py for easier handling of related variables as fields.
- Included a shell file for easier server execution of the code.

TODO: 1. Code for multiple for-loops is not yet developed.

### 20241021 update

- by changlin
- For-loops are now implemented. You need to specify either rotate_angle_list or backward_distance_list, but not both at the same time. (For negative lists, use spaces, e.g., " -10 -20 -30")
- The backward_distance_list field specifies the scaling factor, such as 0.5, 1.0, 1.5.
- Both prompt_question and prompt_diffusion default to None; you must specify values, and they cannot both be None or both not None.
- The input for vlm_diffusion_pipeline.py now includes original_input_img_path to read the initial image; if using VLM, it uses the initial image to obtain the prompt.
- Added the optimize_num_iters field, which sets the number of optimization iterations during Flash3D post-processing. This can currently be ignored.
- Note: Modifications were made near line 40 of generator_test.py, commenting out statements like self.model_cfg_path to adapt to local machine execution; feel free to modify as needed.
