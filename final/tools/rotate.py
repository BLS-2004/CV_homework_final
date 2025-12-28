import cv2

# 直接修改这里的文件名
input_file, output_file = "2.1.png", "2.2.png"

# 一行代码完成旋转和保存
cv2.imwrite(output_file, cv2.rotate(cv2.imread(input_file), cv2.ROTATE_90_CLOCKWISE))