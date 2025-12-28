import cv2
import os

# 使用OpenCV库打印图片尺寸
for filename in ['0.png', '1.png', '2.png']:
    if os.path.exists(filename):
        img = cv2.imread(filename)
        if img is not None:
            height, width = img.shape[:2]
            print(f"{filename}: {width}x{height} 像素")
        else:
            print(f"{filename}: 无法读取图片")
    else:
        print(f"{filename}: 文件不存在")