import cv2
import numpy as np
import math

def my_bilinear(src, scale):
    #########################
    # TODO                  #
    # my_bilinear 완성      #
    #########################
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst))

    # bilinear interpolation 적용
    dst[0, 0] = src[0, 0]
    dst[0, w_dst-1] = src[0, w-1]
    dst[h_dst-1, 0] = src[h-1, 0]
    dst[h_dst-1, w_dst-1] = src[h-1, w-1]
    #src * scale = dst
    #src = dst/scale
    for row in range(h_dst):
        for col in range(w_dst):
            m = math.floor(row / scale) #src에서의 픽셀 위치 y값
            n = math.floor(col / scale) #src에서의 픽셀 위치 x값
            s = (col / scale) - n
            t = (row / scale) - m

            if((row/scale).is_integer() & (col/scale).is_integer()) : # 정수일 때
                dst[row,col] = src[int(row / scale), int(col / scale)]

            elif ((m < h - 1) & (n < w - 1)):
                dst[row, col] = (1 - s) * (1 - t) * src[m, n] + s * (1 - t) * src[m, n + 1] + (1 - s) * t * src[
                    m + 1, n] + s * t * src[m + 1, n + 1]

            elif ((m == h - 1) & (n < w - 1)) :
                dst[row, col] = (1 - s) * (1 - t) * src[m, n] + s * (1 - t) * src[m, n + 1] + (1 - s) * t * src[
                    m, n] + s * t * src[m, n + 1]
            elif ((n == w - 1) & (m < h - 1)) :
                dst[row, col] = (1 - s) * (1 - t) * src[m, n] + s * (1 - t) * src[m, n] + (1 - s) * t * src[
                    m + 1, n] + s * t * src[m + 1, n]
            elif ((m == h - 1) & (n == w - 1)):
                dst[row, col] = (1 - s) * (1 - t) * src[m, n] + s * (1 - t) * src[m, n ] + (1 - s) * t * src[
                    m, n] + s * t * src[m, n]

    #print(dst)
    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    scale = 1/2
    #이미지 크기 1/2배로 변경
    my_dst_mini = my_bilinear(src, scale)
    my_dst_mini = my_dst_mini.astype(np.uint8)

    #이미지 크기 2배로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst = my_bilinear(my_dst_mini, 1/scale)
    my_dst = my_dst.astype(np.uint8)

    scale = 1 / 7
    # 이미지 크기 1/7배로 변경
    my_dst_mini_7 = my_bilinear(src, scale)
    my_dst_mini_7 = my_dst_mini_7.astype(np.uint8)

    # 이미지 크기 7배로 변경(Lena.png 이미지의 shape는 (512, 512))
    my_dst_7 = my_bilinear(my_dst_mini_7, 1 / scale)
    my_dst_7 = my_dst_7.astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini 1/2', my_dst_mini)
    cv2.imshow('my bilinear 1/2', my_dst)

    cv2.imshow('my bilinear mini 1/7', my_dst_mini_7)
    cv2.imshow('my bilinear 1/7', my_dst_7)

    cv2.waitKey()
    cv2.destroyAllWindows()


