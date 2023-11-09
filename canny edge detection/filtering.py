import numpy as np
from math import exp
import math

def my_filtering(src, mask, pad_type):
    (m_h,m_w) = mask.shape
    (h, w) = src.shape  # 흑백이미지
    src_pad = my_padding(src, (m_h // 2, m_w // 2), pad_type)
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
def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w), dtype=float)
    pad_img[p_h:p_h + h, p_w:p_w + w] = src  # p_h에서부터 p_h+h전까지, p_w에서부터 p_w+w전까지 scr넣기

    if pad_type == 'repetition':
        print('repetition padding')
        #########################################################
        # TODO                                                  #
        # repetition padding 완성                                #
        #########################################################
        # up
        pad_img[:p_h,p_w : w + p_w] = src[0, :]
        # down
        pad_img[p_h + h:, p_w:w + p_w] = src[h - 1, :]
        # left
        pad_img[:,:p_w] = pad_img[:,p_w:p_w+1]
        # right
        pad_img[:,p_w+w:] = pad_img[:,p_w+w-1:p_w+w]

    else:
        print('zero padding')

    return pad_img

def get_DoG_filter(fsize, sigma):
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################
    y, x = np.mgrid[-fsize//2+1 : fsize//2 + 1, -fsize//2+1 : fsize//2 + 1]
    DoG_x = (-x/sigma**2) * np.exp(-((x**2 + y**2)/(2 * sigma**2)))
    DoG_y = (-y/sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))

    DoG_x = DoG_x - (DoG_x.sum()/fsize**2)
    DoG_y = DoG_y - (DoG_y.sum()/fsize**2)

    return DoG_x, DoG_y
