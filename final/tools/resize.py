import cv2

# 读取1.png获取目标尺寸
img_target = cv2.imread(r'../1.png')
if img_target is None:
    print("错误: 无法读取img_target")
    exit()

# 获取目标高度和宽度（注意：OpenCV的shape顺序是(height, width)）
target_height, target_width = img_target.shape[:2]

# 读取0.png
img = cv2.imread(r'../images/tutorial/3.png')
if img is None:
    print("错误: 无法读取img")
    exit()

# 调整0.png大小
resized_img0 = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

# 保存调整后的0.png
cv2.imwrite(r'../images/tutorial/3.png', resized_img0)
print(f"img已调整为 {target_width}x{target_height} 像素")
