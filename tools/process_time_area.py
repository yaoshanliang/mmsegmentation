import cv2
import numpy as np
import os


def fill_region_with_specific_color(image, region_position, specific_point):
    x1, y1, x2, y2 = region_position
    sp_x, sp_y = specific_point
    
    # 获取指定点的像素值
    specific_color = image[sp_y, sp_x]
    
    # 将指定区域内的所有像素改为specific_color
    image[y1:y2, x1:x2] = specific_color
    
    return image


# 加载图片
image_path = '/home/shanliang/workspace/code/mmsegmentation/output/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_lars-512x1024/vis_data/vis_image/'

new_image_folder = '/home/shanliang/workspace/code/mmsegmentation/output/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_lars-512x1024/vis_data/vis_image_new/'




# 指定区域的位置信息，需要根据实际情况调整
region_position = [890, 140, 1290, 185]   # [x1, y1, x2, y2]

# 指定点的位置信息
specific_point = [1080, 135]  # [x, y]ss

# 处理图片
images = os.listdir(image_path)
for image_name in images:
    if image_name.startswith('davimar'):
        original_image = cv2.imread(os.path.join(image_path, image_name))
        processed_image = fill_region_with_specific_color(original_image, region_position, specific_point)

        # 显示结果
        cv2.imwrite(os.path.join(new_image_folder, image_name), processed_image)
