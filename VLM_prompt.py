from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

def get_VLM():
    """
    https://huggingface.co/qresearch/llama-3.1-8B-vision-378

    Get the VLM model and tokenizer from the Hugging Face model hub.

    Returns:
    model: VLM model
    tokenizer: VLM tokenizer
    """

    model = AutoModelForCausalLM.from_pretrained(
        "qresearch/llama-3.1-8B-vision-378",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained("qresearch/llama-3.1-8B-vision-378", use_fast=True,)
    return model, tokenizer

def get_prompt(image, vlm_model, vlm_tokenizer,
               question: str = "Briefly describe the image",
               max_tokens: int = 100,
               temperature: float = 0.3):
    """
    https://huggingface.co/qresearch/llama-3.1-8B-vision-378

    Get the prompt for inpainting from the VLM model.

    Args:
    image: PIL.Image
        Image to generate prompt from
    vlm_model: VLM model
    vlm_tokenizer: VLM tokenizer

    Returns:
    prompt: str
        Prompt for inpainting
    """

    prompt = vlm_model.answer_question(
        image, question, vlm_tokenizer, max_new_tokens=max_tokens, do_sample=True, temperature=temperature
    )

    return prompt

if __name__ == '__main__':
    img_folder = './data/input'
    vlm_model, vlm_tokenizer = get_VLM()
    img_prompt_mapping = {}

    prompt_question = 'Describe the content of given image briefly.'
    print(f'prompt:{prompt_question}')
    # 遍历文件夹中的所有图片文件
    for img_file in tqdm(os.listdir(img_folder), desc="Processing Images"):
        img_path = os.path.join(img_folder, img_file)

        if img_file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img = Image.open(img_path).convert("RGB")

            prompt = get_prompt(img, vlm_model, vlm_tokenizer, question=prompt_question)
            img_prompt_mapping[img_file] = prompt

    output_file = img_folder + '/image_prompt_mapping.json'
    with open(output_file, 'w') as f:
        json.dump(img_prompt_mapping, f, indent=4)

    # 清理 GPU 缓存
    del vlm_model, vlm_tokenizer
    torch.cuda.empty_cache()

    json_file_path = output_file
    with open(json_file_path, 'r') as f:
        img_prompt_mapping = json.load(f)
    for img_file, prompt in img_prompt_mapping.items():
        print(f"Image: {img_file} -> Description: {prompt}")