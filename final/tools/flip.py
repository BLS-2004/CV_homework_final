import cv2
import sys

input_file, output_file = "../2.png", "../4.png"
# 一行代码版本
cv2.imwrite(output_file, cv2.flip(cv2.imread(input_file), 1))