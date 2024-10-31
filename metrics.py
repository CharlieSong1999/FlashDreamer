import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from pytorch_fid import fid_score
from tqdm import tqdm
from PIL import Image
import torch
import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def cal_fid_score(dataset1_path, dataset2_path, batch_size=50,
                        device='cuda' if torch.cuda.is_available() else 'cpu'):
    fid_value = fid_score.calculate_fid_given_paths( # resize to 299x299
        [dataset1_path, dataset2_path],
        batch_size=batch_size,
        device=device,
        dims=2048
    )

    return fid_value

def calculate_clip_score(image_path, text_description, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=[text_description],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

    # 归一化特征向量
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # 计算余弦相似度
    similarity = (image_features @ text_features.T).cpu().item()

    return similarity

def calculate_average_clip_score(description_dict, image_folder_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 加载 CLIP 模型和处理器
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    scores = []
    # 遍历文件夹中的每个图像文件
    for image_filename in os.listdir(image_folder_path):
        image_key = image_filename.split('_')[0] + '_4.jpg'
        image_path = os.path.join(image_folder_path, image_filename)
        # 确保图像文件在描述字典中有对应的描述
        if image_key in description_dict:
            text_description = description_dict[image_key]
            score = calculate_clip_score(image_path, text_description, model, processor, device)
            scores.append(score)
        else:
            print(f"Warning: No description found for {image_filename}, skipping.")

    # 计算所有得分的平均值
    if scores:
        average_score = sum(scores) / len(scores)
    else:
        average_score = 0.0

    return average_score

def calculate_scores(src_path, method_name):
    fid_scores = []
    clip_scores = []
    src_path_gt = './data/gt'
    angle_folders = [-30, -20, -10, 10, 20, 30]

    # 计算 FID 分数
    for angle in angle_folders:
        dataset1_path = os.path.join(src_path_gt, str(angle))
        dataset2_path = os.path.join(src_path, str(angle))
        fid_value = cal_fid_score(dataset1_path, dataset2_path)
        fid_scores.append(f"{fid_value:.2f}")

    with open('./data/input/image_prompt_mapping.json', 'r') as f:
        description_dict = json.load(f)

    # 计算 CLIP scores
    for angle in angle_folders:
        image_folder_path = os.path.join(src_path, str(angle))
        average_clip_score = calculate_average_clip_score(description_dict, image_folder_path)
        clip_scores.append(f"{average_clip_score:.2f}")

    # 按格式输出结果
    result_row = f"{method_name} & " + " & ".join(f"{fid} & {clip}" for fid, clip in zip(fid_scores, clip_scores)) + " \\\\"
    print(result_row)


if __name__ == '__main__':
    calculate_scores('./data/pixelSyn_output/', "PixelSyn")
    calculate_scores('./data/output/', "Ours")
    # src_path_1 = './data/gt/'
    # src_path_2 = './data/pixelSyn_output/'
    # src_path_2 = './data/output/'
    #
    # angle_folders = [10, 20, 30, -10, -20, -30]
    #
    # # # '''FID'''
    # for angle in angle_folders:
    #     dataset1_path = os.path.join(src_path_1, str(angle))
    #     dataset2_path = os.path.join(src_path_2, str(angle))
    #     fid_value = cal_fid_score(dataset1_path, dataset2_path)
    #     print(f"Angle:{angle} - FID: {fid_value}")
    #
    # '''CLIP score'''
    # with open('./data/input/image_prompt_mapping.json', 'r') as f:
    #     description_dict = json.load(f)
    #
    # for angle in angle_folders:
    #     image_folder_path = os.path.join(src_path_2, str(angle))
    #     average_clip_score = calculate_average_clip_score(description_dict, image_folder_path)
    #     print(f"Angle:{angle} - CLIP score: {average_clip_score}")