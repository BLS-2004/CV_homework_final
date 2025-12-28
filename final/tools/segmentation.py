import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
img_path = '../images/danger.png'  # 请根据需要修改图片路径
img = cv2.imread(img_path)
if img is None:
    print(f"错误：无法读取图片 {img_path}，请检查文件路径")
    exit()

height, width = img.shape[:2]

# 预处理 - 增强红色通道
b, g, r = cv2.split(img)
red_enhanced = cv2.subtract(r, cv2.add(g, b) // 2)
red_enhanced = cv2.normalize(red_enhanced, None, 0, 255, cv2.NORM_MINMAX)

# 边缘检测
blurred = cv2.GaussianBlur(red_enhanced, (5, 5), 0)
edges = cv2.Canny(blurred, 30, 100)

# 形态学操作
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=2)

# 轮廓检测
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()

# 在原图上绘制轮廓
if contours:
    # 按面积排序
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

    # 绘制所有显著轮廓
    for i, contour in enumerate(contours_sorted):
        area = cv2.contourArea(contour)
        if area > 100:  # 只绘制面积大于100的轮廓
            # 用不同颜色绘制不同轮廓
            color = (0, 255, 0) if i == 0 else (0, 165, 255)  # 最大轮廓用绿色，其他用橙色
            cv2.drawContours(contour_img, [contour], -1, color, 2)

    # 创建掩码
    mask = np.zeros((height, width), dtype=np.uint8)
    for contour in contours_sorted:
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # 创建分割结果
    green_background = np.zeros_like(img)
    green_background[:] = (0, 255, 0)  # 纯绿色背景

    foreground = cv2.bitwise_and(img, img, mask=mask)
    background_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(green_background, green_background, mask=background_mask)
    result = cv2.add(foreground, background)

    # 保存结果
    cv2.imwrite('../images/danger_result.png', result)

    # 显示轮廓和分割结果
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title(f'检测到{len(contours_sorted)}个轮廓')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('分割结果')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
