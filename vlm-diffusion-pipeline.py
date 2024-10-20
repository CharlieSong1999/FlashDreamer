import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from diffusers.utils import load_image, make_image_grid
import argparse
from matplotlib import pyplot as plt
from transformers import MllamaForConditionalGeneration, AutoProcessor
import cv2

def get_VLM(vlm_model_name ='llama-3.2'):
    """
    https://huggingface.co/qresearch/llama-3.1-8B-vision-378

    Get the VLM model and tokenizer from the Hugging Face model hub.

    Returns:
    model: VLM model
    tokenizer: VLM tokenizer
    """
    if vlm_model_name == 'llama-3.1':
        model = AutoModelForCausalLM.from_pretrained(
            "./llama-3.1-8B-vision-378",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to("cuda")

        tokenizer = AutoTokenizer.from_pretrained("./llama-3.1-8B-vision-378", use_fast=True,)
    elif vlm_model_name == 'llama-3.2':
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer = AutoProcessor.from_pretrained(model_id)
    else:
        raise ValueError(f"Invalid vlm_model: {vlm_model_name}")

    return model, tokenizer

def get_Inpainting_Pipeline(base_model='stable-diffusion'):
    """
    Get the inpainting pipeline from the Hugging Face model hub.

    Args:
    base_model: str, default 'stable-diffusion'
        The base model to use for inpainting. Options are 'stable-diffusion' and 'kandinsky-2-2'.
    
    Returns:
    pipeline: Inpainting pipeline
    """

    if base_model == 'stable-diffusion':
        # https://huggingface.co/stabilityai/stable-diffusion-2-inpainting
        from diffusers import StableDiffusionInpaintPipeline
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
        pipeline.to("cuda")
    elif base_model == 'stable-diffusion-v2':
        from diffusers import StableDiffusionInpaintPipeline
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
        pipeline.to("cuda")
    elif base_model == 'kandinsky-2-2':
        # https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint
        from diffusers import AutoPipelineForInpainting

        pipeline = AutoPipelineForInpainting.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
        )
        pipeline.to("cuda")
    elif base_model == 'stable-diffusion-xl':
        from diffusers import AutoPipelineForInpainting
        pipeline = AutoPipelineForInpainting.from_pretrained(
            "./stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16"
        )
        pipeline.to("cuda")
    else:
        raise ValueError(f"Invalid base_model: {base_model}")
    
    
    pipeline.enable_model_cpu_offload()

    # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
    # pipeline.enable_xformers_memory_efficient_attention()

    return pipeline

def get_image_and_mask(image_path, mask_path):
    """
    Load the image and mask from the given paths.

    Args:
    image_path: str
        Path to the image file
    mask_path: str
        Path to the mask file

    Returns:
    image: torch.Tensor
        Image tensor
    mask: torch.Tensor
        Mask tensor
    """

    image = load_image(image_path)
    mask = load_image(mask_path)

    return image, mask

def inpaint_image(init_image, mask_image, prompt, pipeline, seed=114514):
    """

    Args:
    init_image: torch.Tensor
        Image tensor
    mask_image: torch.Tensor
        Mask tensor (an image with the same size as the image tensor)
    prompt: str
        Prompt for inpainting
    pipeline: Inpainting pipeline
    seed: int, default 114514
        Random seed for inpainting
    
    Returns:
    image: PIL.Image
        Inpainted image
    """

    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]

    return image

def get_prompt(image, vlm_model_name, vlm_model, vlm_tokenizer,
               question: str = "Briefly describe the image",
               max_tokens: int = 256,
               temperature: float = 0.3):
    """
    
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
    if vlm_model_name == 'llama-3.1':
        # https://huggingface.co/qresearch/llama-3.1-8B-vision-378
        prompt = vlm_model.answer_question(
            image, question, vlm_tokenizer, max_new_tokens=max_tokens, do_sample=True, temperature=temperature
        )

        return prompt
    elif vlm_model_name == 'llama-3.2':
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]}
        ]
        input_text = vlm_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = vlm_tokenizer(image, input_text, return_tensors="pt").to(vlm_model.device)

        output = vlm_model.generate(**inputs, max_new_tokens=max_tokens)

        prompt = vlm_tokenizer.decode(output[0])

        print(f'Ori prompt: {prompt}')

        if '<|end_header_id|>' in prompt:
            prompt = prompt.split('<|end_header_id|>')[-1].strip()
        
        if '<|eot_id|>' in prompt:
            prompt = prompt.split('<|eot_id|>')[0].strip()

        print(f'New prompt: {prompt}')
    
        return prompt

def main(image_path, mask_path, prompt_question, base_model='stable-diffusion', vlm_model_name='llama-3.2', prompt_diffusion=None):
    """
    Main function to inpaint an image using the VLM model.

    Args:
    image_path: str
        Path to the image file
    mask_path: str
        Path to the mask file
    prompt_question: str
        Prompt question for inpainting
    base_model: str, default 'stable-diffusion'
        The base model to use for inpainting. Options are 'stable-diffusion' and 'kandinsky-2-2'.
    """

    # Load the image and mask
    image, mask = get_image_and_mask(image_path, mask_path)

    # Get diffusion prompt
    if not prompt_diffusion:
        # Get the VLM model and tokenizer
        vlm_model, vlm_tokenizer = get_VLM(vlm_model_name)
        print(f'Getting diffusion prompt using question: {prompt_question}...')
        prompt = get_prompt(image, vlm_model_name, vlm_model, vlm_tokenizer, question=prompt_question)

        del vlm_model, vlm_tokenizer
        torch.cuda.empty_cache()
    else:
        prompt = prompt_diffusion

    # Get the inpainting pipeline
    pipeline = get_Inpainting_Pipeline(base_model)

    # Inpaint the image
    print(f'Inpainting image using prompt: {prompt}')
    inpainted_image = inpaint_image(image, mask, prompt, pipeline)

    # Display the inpainted image
    grid_img = make_image_grid([image, mask, inpainted_image], rows=1, cols=3)
    plt.imshow(grid_img)
    plt.axis('off')
    plt.show()

    inpainted_image.save('./output/inpainting_result.png')

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--image_path", type=str, default='./imgs/room-512.jpg', help="Path to the image file")
    argparser.add_argument("--mask_path", type=str, default='./imgs/room-mask-512.webp', help="Path to the mask file")
    argparser.add_argument("--prompt_question", type=str, default='Briefly describe the image in 30 words, ignore the blurred part', help="Prompt question for inpainting")
    argparser.add_argument("--prompt_diffusion", type=str, default=None, help="Direct prompt for diffusion")
    argparser.add_argument("--base_model", type=str, default='stable-diffusion-xl', help="Base model for inpainting")
    argparser.add_argument("--vlm_model", type=str, default='llama-3.1', help="VLM model to use")
    args = argparser.parse_args()

    main(args.image_path, args.mask_path, args.prompt_question, args.base_model, args.vlm_model, args.prompt_diffusion)