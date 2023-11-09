import numpy as np
from math import exp
import math
from my_library import padding

def my_filtering(src, mask, pad_type):
    (m_h,m_w) = mask.shape
    (h, w) = src.shape  # 흑백이미지
    src_pad = padding.my_padding(src, (m_h // 2, m_w // 2), pad_type)
    dst = np.zeros((h, w), dtype=float)

    for row in range(h):
       for col in range(w):
           sum = np.sum(src_pad[row:row+m_h, col:col+m_w] * mask)
           if sum > 255 :
               sum = 255
           elif sum < 0 :
               sum = 0
           dst[row, col] = sum

    return dst

