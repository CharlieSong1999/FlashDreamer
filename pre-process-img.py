import cv2
from matplotlib import pyplot as plt
import argparse

if __name__ == '__main__':

    """
    Pre-process the image and mask before inpainting.

    It will take the flash3dmasked_rendered.png to create the mask by setting every pixel values > 0 to 0 and every pixel values == 0 to 255.
    The mask will be resized to the output_height and output_width.
    It will take the flash3drender_test.png to resize the image to the output_height and output_width.

    
    """

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--image_path", type=str, default='./flash3drender_test.png', help="Path to the image file")
    argparser.add_argument("--mask_path", type=str, default='./flash3dmasked_rendered.png', help="Path to the mask file")
    argparser.add_argument("--out_resize_path", type=str, default='./flash3drender_test_resized.png', help="Path to the mask file")
    argparser.add_argument("--out_mask_resized_path", type=str, default='./flash3drender_test_mask_resized.png', help="Path to the mask file")
    argparser.add_argument("--output_height", type=int, default=1024, help="Height of the output image")
    argparser.add_argument("--output_width", type=int, default=1024, help="Width of the output image")
    args = argparser.parse_args()

    # Load the mask image
    mask_img = cv2.imread(args.mask_path, cv2.IMREAD_COLOR)

    # get mask
    mask = mask_img.copy()
    mask[mask_img == 0] = 255
    mask[mask_img > 0] = 0

    # Load the image
    img = cv2.imread(args.image_path, cv2.IMREAD_COLOR)

    # Resize the image
    img = cv2.resize(img, (args.output_height, args.output_width))
    mask = cv2.resize(mask, (args.output_height, args.output_width))

    # Save the resized image
    cv2.imwrite(args.out_resize_path, img)
    cv2.imwrite(args.out_mask_resized_path, mask)