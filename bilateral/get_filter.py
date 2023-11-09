import numpy as np
from math import exp

def my_get_Gaussian2D_mask(msize, sigma=1):
    #########################################
    # ToDo
    # 2D gaussian filter 만들기
    #########################################
    y, x = np.mgrid[-msize//2+1 : msize//2 + 1, -msize//2+1 : msize//2 + 1,]

    temp = np.zeros((msize, msize))
    temp = -(y * y + x * x) / (2 * sigma)
    # 2차 gaussian mask 생성
    π = 3.14159265359
    for i in range(-msize//2+1, msize//2+1):
        for j in range(-msize//2+1, msize//2+1):
            temp[i, j] = exp(temp[i, j]) * 1/(2 * π)
    temp = temp / np.sum(temp)

    gaus2D = temp
    return gaus2D