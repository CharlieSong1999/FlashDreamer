import cv2
import os

current_directory = '/scratch/kf09/lz1278/ANU-COMP8536-2024s2-main/flash3d-cache'
# 指定图片文件夹路径和视频输出路径
image_folder = current_directory+'/rotate_demo'  # 替换为你存储 PNG 图片的文件夹路径
video_name = current_directory+'/rotate_demo/output_video.mp4'  # 输出视频的文件名

# 获取文件夹中的所有图片文件名，按文件名中的数字顺序排列
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

# 使用 sorted() 按文件名中的数字排序
images = sorted(images, key=lambda x: int(x.split('_')[0]))

# 读取第一张图片来获取帧的宽度和高度
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 定义视频编码器和帧率（这里使用 MJPG 编码，帧率设为 10）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以使用 'XVID' 或其他编码格式
video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

# 将每张图片写入视频文件
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# 释放视频写入对象
video.release()

print(f'视频已生成并保存为 {video_name}')
