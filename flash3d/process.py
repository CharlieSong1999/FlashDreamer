from torchvision import transforms
from PIL import Image
import sys
import os
current_directory = os.path.dirname(os.path.abspath(__file__))

# 打开图片
image = Image.open(current_directory+"/img1.png")  # 替换为你的图片路径

# 定义变换: 先调整大小为 384x640，中心裁剪为 320x576，最后再填充 32 像素
transform = transforms.Compose([
    transforms.Resize((384, 640)),  # 先调整大小为 (height, width)
    transforms.CenterCrop((320, 576)),  # 再中心裁剪为 (height, width)
    transforms.Pad(padding=32, fill=(0, 0, 0))  # 最后添加 32 像素的填充，填充颜色为黑色
])

# 应用变换
transformed_image = transform(image)

# 保存结果
transformed_image.save(current_directory+"/diffusion.png")
