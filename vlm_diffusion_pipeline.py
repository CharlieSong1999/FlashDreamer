import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from diffusers.utils import load_image, make_image_grid
import argparse
from matplotlib import pyplot as plt
from torchvision.transforms.functional import resize, pil_to_tensor
import json

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
    pipeline.enable_xformers_memory_efficient_attention()

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

def inpaint_image(init_image, mask_image, prompt, pipeline, seed=114514, strength=0.6, negative_prompt=None):
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

    if negative_prompt is not None:
        image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator, strength=strength, negative_prompt=negative_prompt).images[0]
    else:
        image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator, strength=strength).images[0]

    return image

def get_prompt(image, vlm_model, vlm_tokenizer,
               question: str = "Briefly describe the image",
               max_tokens: int = 128,
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

def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    Overlay the mask on the original image with a given color and transparency.

    Args:
    image: PIL.Image
        Original image.
    mask: PIL.Image
        Binary mask image.
    color: tuple
        RGB color for the mask overlay (default is red).
    alpha: float
        Transparency level of the mask overlay (0 = fully transparent, 1 = fully opaque).

    Returns:
    PIL.Image: Image with mask overlay.
    """
    image = image.convert("RGBA")
    mask = mask.convert("L")

    overlay = Image.new("RGBA", image.size, color + (0,))
    mask_color = Image.new("RGBA", image.size, color + (int(alpha * 255),))
    overlay = Image.composite(mask_color, overlay, mask)
    blended_image = Image.alpha_composite(image, overlay)

    return blended_image.convert("RGB")

def visualize_inpainting_comparison(image, mask, inpainted_image):
    """
    Visualize the original image, mask overlay on the original image, and the inpainted result.

    Args:
    image: PIL.Image
        Original image resized to 1024x1024.
    mask: PIL.Image
        Mask image resized to 1024x1024.
    inpainted_image: PIL.Image
        The inpainted image generated by the pipeline (1024x1024).
    """
    image_with_mask = overlay_mask_on_image(image, mask)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image)
    axs[0].set_title("Original Image (Resized)")
    axs[0].axis("off")

    axs[1].imshow(image_with_mask)
    axs[1].set_title("Image with Mask")
    axs[1].axis("off")

    axs[2].imshow(inpainted_image)
    axs[2].set_title("Inpainted Image")
    axs[2].axis("off")

    # plt.savefig('./compare_inpaint.jpg')
    # plt.show()
    return fig

def get_image_embedding(image, model_name='blip-2'):

    if model_name == 'blip-2':
        from transformers import AutoProcessor, Blip2VisionModelWithProjection
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")
        model = Blip2VisionModelWithProjection.from_pretrained(
            "Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16
        ).to("cuda")

        inputs = processor(images=image, return_tensors="pt").to('cuda', torch.float16)
        outputs = model(**inputs)

        image_embeds = outputs.last_hidden_state

        del model, processor
        torch.cuda.empty_cache()

    print('Image embedding shape:', image_embeds.shape)

    return image_embeds

def main(original_input_img_path, image_path, mask_path, prompt_question, base_model='stable-diffusion', prompt_diffusion=None, index=0, strength=0.6,
         negative_prompt = None):
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

    # load the original image
    original_input_img = Image.open(original_input_img_path)

    # Load the image and mask
    image, mask = get_image_and_mask(image_path, mask_path)

    # print('before resize', image.size, mask.size)
    # image = resize(image, [512, 512])
    # mask = resize(mask, [512, 512])
    # print('after resize', image.size, mask.size)

    # Get the prompt
    if prompt_diffusion is None:
        try: # Load from predefined prompt
            json_file_path = os.path.dirname(original_input_img_path) + '/image_prompt_mapping.json'
            with open(json_file_path, 'r') as f:
                img_prompt_mapping = json.load(f)
            prompt = img_prompt_mapping[os.path.basename(original_input_img_path)]
            print('Load prompt for diffusion from the local.')
        except Exception as e:
            # Get the VLM model and tokenizer
            vlm_model, vlm_tokenizer = get_VLM()
            prompt = get_prompt(original_input_img, vlm_model, vlm_tokenizer, question=prompt_question)
            del vlm_model, vlm_tokenizer
            torch.cuda.empty_cache()
            print('Generate prompt with VLM.')
    else:
        prompt = prompt_diffusion
        print('Use the given prompt.')


    # Get the inpainting pipeline
    prompt = 'Please generate a real-life picture. According to the description:' + prompt
    print('Prompt:', prompt)

    pipeline = get_Inpainting_Pipeline(base_model)
    seed = 10241024
    generator = torch.Generator("cuda").manual_seed(seed)
    inpainted_image = \
        pipeline(prompt=prompt, image=image, mask_image=mask, strength=1,
                 negative_prompt=None, num_inference_steps=100).images[0]

    fig = visualize_inpainting_comparison(image, mask, inpainted_image)
    fig.savefig(f'./imgs/inpainting_{index}.jpg')

    del pipeline
    torch.cuda.empty_cache()

    # inpainted_image.save(f'./ly_test_imgs/inpainting_{index}_{prompt}.jpg')

    return inpainted_image

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--image_path", type=str, default='./imgs/room-512.jpg', help="Path to the image file")
    argparser.add_argument("--mask_path", type=str, default='./imgs/room-mask-512.webp', help="Path to the mask file")
    argparser.add_argument("--prompt_question", type=str, default='Briefly describe the image', help="Prompt question for inpainting")
    argparser.add_argument("--prompt_diffusion", type=str, default=None, help="Direct prompt for diffusion")
    argparser.add_argument("--base_model", type=str, default='stable-diffusion-xl', help="Base model for inpainting")
    args = argparser.parse_args()

    main(args.image_path, args.mask_path, args.prompt_question, args.base_model, args.prompt_diffusion)