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

