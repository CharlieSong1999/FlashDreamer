import os
import shutil

# 设置源目录路径
source_dir = './data/exp_set'
# 设置目标目录路径
input_dir = './data/input'
gt_dir = './data/gt'

# 创建目标目录
os.makedirs(input_dir, exist_ok=True)
os.makedirs(gt_dir, exist_ok=True)

for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)

    # Check if the item is a directory
    if os.path.isdir(folder_path):
        # Get a sorted list of images (assuming .jpg files)
        images = sorted([img for img in os.listdir(folder_path) if img.endswith('.jpg')])

        # Rename each image in the folder
        for i, img_name in enumerate(images):
            old_path = os.path.join(folder_path, img_name)
            new_name = f"{folder_name}_{i + 1}.jpg"  # Rename format: <folder_name>_<index>.jpg
            new_path = os.path.join(folder_path, new_name)

            # Rename the image
            os.rename(old_path, new_path)
            #print(f"Renamed '{img_name}' to '{new_name}' in folder '{folder_name}'")


print(len(os.listdir(source_dir)))
# 遍历每个编号文件夹
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path):
        #continue
        # 获取文件夹内的图片文件并排序
        images = sorted([img for img in os.listdir(folder_path) if img.endswith('.jpg')])

        # 确保文件夹中至少有一张中间编号的图片
        if len(images) % 2 == 1:
            # 计算中间图片的索引
            mid_index = len(images) // 2
            gt_image = images[mid_index]
            gt_image_path = os.path.join(folder_path, gt_image)

            # 将中间编号图片复制到 input 文件夹
            shutil.copy(gt_image_path, os.path.join(input_dir, gt_image))

            # 处理旋转角度图片
            for i, image in enumerate(images):
                if i != mid_index:  # 排除中间的 GT 图片
                    angle = (i - mid_index) * 10  # 根据索引计算旋转角度
                    angle_dir = os.path.join(gt_dir, f'{angle}')
                    os.makedirs(angle_dir, exist_ok=True)

                    # 复制旋转图片到对应角度的子文件夹
                    shutil.copy(os.path.join(folder_path, image), os.path.join(angle_dir, image))
        else:
            print(folder_path)

print("文件分类完成！")
