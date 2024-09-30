python ./flash3d/generator.py
python pre-process-img.py --output_height 512 --output_width 512
python .\vlm-diffusion-pipeline.py --image_path 'flash3drender_test_resized.png' --mask_path 'flash3drender_test_mask_resized.png' --base_model 'stable-diffusion-xl'