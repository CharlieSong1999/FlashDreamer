	
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

# Set the path to the input image and the output directory, the input image is the very first image of the whole pipeline
intial_img_path = './flash3d/frame000652.jpg'
output_path = './flash3d-output'
current_directory = './flash3d-cache'

flash3dreconstructor = Flash3DReconstructor()