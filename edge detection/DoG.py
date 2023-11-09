import cv2
import numpy as np
from math import exp


# library add
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.filtering import my_filtering

def get_DoG_filter(fsize, sigma):
    ###################################################
    # TODO                                            #
    # DoG mask 완성                                    #
    ###################################################
    y, x = np.mgrid[-fsize//2+1 : fsize//2 + 1, -fsize//2+1 : fsize//2 + 1]
    temp = np.zeros((fsize, fsize))
    temp = -(y * y + x * x) / (2 * sigma* sigma)
    # 2차 gaussian mask 생성
    for i in range(-fsize//2+1, fsize//2+1):
        for j in range(-fsize//2+1, fsize//2+1):
            temp[i, j] = exp(temp[i, j]) * -1/(sigma * sigma)
    tempx = np.zeros((fsize, fsize))
    tempy = np.zeros((fsize, fsize))
    tempx = temp
    tempy = temp
    DoG_x =tempx * x
    DoG_y =tempy * y
    return DoG_x, DoG_y

def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    DoG_x, DoG_y = get_DoG_filter(fsize=3, sigma=1)

    ###################################################
    # TODO                                            #
    # DoG mask sigma값 조절해서 mask 만들기              #
    ###################################################
    # DoG_x, DoG_y filter 확인
    x, y = get_DoG_filter(fsize=256, sigma=50)
    x = ((x - np.min(x))/np.max(x - np.min(x)) * 255).astype(np.uint8)
    y = ((y - np.min(y))/np.max(y - np.min(y)) * 255).astype(np.uint8)
    dst_x = my_filtering(src, DoG_x, 'zero')
    dst_y = my_filtering(src, DoG_y, 'zero')

    ###################################################
    # TODO                                            #
    # dst_x, dst_y 를 사용하여 magnitude 계산            #
    ###################################################
    dst = ((dst_x ** 2) + (dst_y ** 2)) ** 0.5
    cv2.imshow('DoG_x filter', x)
    cv2.imshow('DoG_y filter', y)
    cv2.imshow('dst_x', dst_x/255)
    cv2.imshow('dst_y', dst_y/255)
    cv2.imshow('dst', dst/255)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

