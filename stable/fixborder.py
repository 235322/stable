import cv2
import numpy as np

# 修复由于变换导致的边界问题
def fix_border(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.1)
    return cv2.warpAffine(frame, T, (s[1], s[0]))
