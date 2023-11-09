import cv2
import numpy as np
import math

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from my_library.filtering import my_filtering
from my_library.filtering import get_DoG_filter

# low-pass filter를 적용 후 high-pass filter적용
def apply_lowNhigh_pass_filter(src, fsize, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering을 해도 되고, DoG를 이용해 한번에 해도 됨

    ###########################################
    # TODO                                    #
    # apply_lowNhigh_pass_filter 완성          #
    # Ix와 Iy 구하기                            #
    ###########################################
    DoG_x, DoG_y = get_DoG_filter(fsize=fsize, sigma=sigma)
    Ix = my_filtering(src, DoG_x, 'zero')
    Iy = my_filtering(src, DoG_y, 'zero')
    return Ix, Iy

# Ix와 Iy의 magnitude를 구함
def calcMagnitude(Ix, Iy):
    ###########################################
    # TODO                                    #
    # calcMagnitude 완성                      #
    # magnitude : ix와 iy의 magnitude         #
    ###########################################
    # Ix와 Iy의 magnitude를 계산
    magnitude = np.sqrt((Ix ** 2) + (Iy ** 2))
    return magnitude

# Ix와 Iy의 angle을 구함
def calcAngle(Ix, Iy):
    ###################################################
    # TODO                                            #
    # calcAngle 완성                                   #
    # angle     : ix와 iy의 angle                      #
    # e         : 0으로 나눠지는 경우가 있는 경우 방지용     #
    # np.arctan 사용하기(np.arctan2 사용하지 말기)        #
    ###################################################
    e = 1E-6
    angle = np.rad2deg(np.arctan(Iy/(Ix+e)))
    return angle

# non-maximum supression 수행
def non_maximum_supression(magnitude, angle):
    ####################################################################################
    # TODO                                                                             #
    # non_maximum_supression 완성                                                       #
    # largest_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)           #
    ####################################################################################
    h, w = magnitude.shape
    largest_magnitude = np.zeros((h, w))
    for row in range(1, h-1) :
        for col in range(1, w-1) :
            degree = angle[row, col]

            if 0<= degree and degree < 45 :
                rate = np.tan(np.deg2rad(degree))
                left_magnitude = (rate) * magnitude[row-1, col-1] + (1-rate) * magnitude[row, col-1]
                right_magnitude = (rate) * magnitude[row+1,col+1] + (1-rate) * magnitude[row, col+1]
                if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude) :
                    largest_magnitude[row, col] = magnitude[row, col]

            elif 45 <= degree and degree <= 90 :
                rate = 1 / np.tan(np.deg2rad(degree))
                up_magnitude = (1-rate) * magnitude[row-1, col] + rate * magnitude[row-1, col-1]
                down_magnitude = (1-rate) * magnitude[row+1, col] + rate * magnitude[row+1, col+1]
                if magnitude[row, col] == max(up_magnitude, magnitude[row, col], down_magnitude) :
                    largest_magnitude[row, col] = magnitude[row, col]

            elif -45 <= degree and degree < 0:
                rate = -np.tan(np.deg2rad(degree))
                left_magnitude = (rate) * magnitude[row + 1, col - 1] + (1 - rate) * magnitude[row, col - 1]
                right_magnitude = (rate) * magnitude[row - 1, col + 1] + (1 - rate) * magnitude[row, col + 1]
                if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            elif -90 <= degree and degree <= -45 :
                rate = -1 / np.tan(np.deg2rad(degree))
                up_magnitude = (1-rate) * magnitude[row-1, col] + rate * magnitude[row-1, col+1]
                down_magnitude = (1-rate) * magnitude[row+1, col] + rate * magnitude[row+1, col-1]
                if magnitude[row, col] == max(up_magnitude, magnitude[row, col], down_magnitude) :
                    largest_magnitude[row, col] = magnitude[row, col]

    return largest_magnitude
# double_thresholding 수행
def double_thresholding(src):
    dst = src.copy()

    #dst => 0 ~ 255
    dst -= dst.min()
    dst /= dst.max()
    dst *= 255
    dst = dst.astype(np.uint8)

    (h, w) = dst.shape
    high_threshold_value, _ = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
    # high threshold value는 내장함수(otsu방식 이용)를 사용하여 구하고
    # low threshold값은 (high threshold * 0.4)로 구한다
    low_threshold_value = high_threshold_value * 0.4

    ######################################################
    # TODO                                               #
    # double_thresholding 완성                            #
    # dst     : double threshold 실행 결과 이미지           #
    ######################################################

    for row in range(h) :
        for col in range(w) :
            if dst[row, col] >= high_threshold_value :
                dst[row, col] = 255
            elif dst[row,col] < low_threshold_value :
                dst[row, col] = 0
            else :
                weak_edge = []
                weak_edge.append((row, col))
                search_weak_edge(dst, weak_edge, high_threshold_value, low_threshold_value)
                if calssify_edge(dst, weak_edge, high_threshold_value) :
                    for idx in range(len(weak_edge)) :
                        (r,c) = weak_edge[idx]
                        dst[r,c] = 255
                else :
                    for idx in range(len(weak_edge)) :
                        (r,c) = weak_edge[idx]
                        dst[r,c] = 0
    return dst
def search_weak_edge(dst, edges,high_threshold_value, low_threshold_value) :
    (row, col) = edges[-1]
    for i in range(-1,2) :
        for j in range(-1,2) :
            if dst[row+i, col+j] < high_threshold_value and dst[row+i, col+j] >= low_threshold_value :
                if edges.count((row+i, col+j)) < 1 :
                    edges.append((row+i, col+j))
                    search_weak_edge(dst, edges, high_threshold_value, low_threshold_value)

def calssify_edge(dst, weak_edge, high_threshold_value) :
    for idx in range(len(weak_edge)) :
        (row, col) = weak_edge[idx]
        value = np.max(dst[row-1:row+2, col-1:col+2])
        if value >= high_threshold_value :
            return True

def my_canny_edge_detection(src, fsize=3, sigma=1):
    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering
    # DoG 를 사용하여 1번 filtering
    Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma)

    # Ix와 Iy 시각화를 위해 임시로 Ix_t와 Iy_t 만들기
    Ix_t = np.abs(Ix)
    Iy_t = np.abs(Iy)
    Ix_t = Ix_t / Ix_t.max()
    Iy_t = Iy_t / Iy_t.max()

    cv2.imshow("Ix", Ix_t)
    cv2.imshow("Iy", Iy_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    angle = calcAngle(Ix, Iy)

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    magnitude_t = magnitude
    magnitude_t = magnitude_t / magnitude_t.max()
    cv2.imshow("magnitude", magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # non-maximum suppression 수행
    largest_magnitude = non_maximum_supression(magnitude, angle)

    # magnitude 시각화를 위해 임시로 magnitude_t 만들기
    largest_magnitude_t = largest_magnitude
    largest_magnitude_t = largest_magnitude_t / largest_magnitude_t.max()
    cv2.imshow("largest_magnitude", largest_magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # double thresholding 수행
    dst = double_thresholding(largest_magnitude)
    return dst

def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_canny_edge_detection(src)
    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()