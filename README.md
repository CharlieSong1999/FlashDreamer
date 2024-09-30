# Quick Start

`python vlm-diffusion-pipeline.py`

# Environment

requirements.txt

# Main idea

目前的 pipeline 包含 (1) 用 llama3.1 生成文字 prompt (2) 根据给定图片、mask图片、以及prompt对图片进行补全。

可以通过设置 --prompt_diffusion 输入参数来越过 llama 生成的promp. 

> 截图里的prompt: The image is a photo taken from a first-person perspective, showing a person walking into a room. The room appears to be a modern office space with a large window and a white couch.

TODO(issues):
1. 目前直接使用llama3.1生成的prompt效果不太理想，可自行尝试
2. mask图片仍需手动生成，目前的pipeline只支持直接读取mask文件(--mask_path 输入参数)
3. 目前huggingface inpaint pipeline似乎只支持512x512的图片，输入其他尺寸会导致输出图片尺寸错误，因此需要手动对图片进行resize，imgs 文件夹中的图片已经resize了.

## 20240930 update

- 添加llama3.2支持，需要指定 --vlm_model 字段
- 添加stable-diffusion-xl 和 stable-diffusion-v2，需要指定 --base_model 字段
- pre-process-img.py 用于将flash3d渲染出来的图片预处理到diffusion接受的格式：512x512 (1024x1024 for stable-diffusion-xl)。他会根据flash3dmasked这个文件，将有像素值的部分覆盖为0，无像素值的部分覆盖为 255。mask和flash3drender_test这两个图片都会被resize。
- flash3d_mask_resize_diffuse.bat 一个windows批处理文件，自动进行整个pipeline （但是在我的机子上，其实最后一行总是找不到文件，推测为windows的路径问题）作为参考。