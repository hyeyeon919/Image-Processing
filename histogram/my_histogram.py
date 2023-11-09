import numpy as np
import cv2
import matplotlib.pyplot as plt

src = cv2.imread('fruits_div3.jpg', cv2.IMREAD_GRAYSCALE) # 원본이미지 값
(h,w) = src.shape

hist = np.zeros((256,), np.int)
def my_calcHist(src) : # 히스토그램 구하기
    for i in range(480) :
        for j in range(512) :
            hist[src[i,j]] = hist[src[i,j]] +1

normalize_hist = np.zeros(256, np.float)
def my_normalize_hist(hist, pixel_num) : #픽셀값으로 나눔
    for i in range(256):
        normalize_hist[i] = hist[i]/pixel_num

my_PDF2CDF_hist = np.zeros(256, np.float)
def my_PDF2CDF(normalize_hist) : #CDF
    my_PDF2CDF_hist[0] = normalize_hist[0]
    for i in range(255):
        my_PDF2CDF_hist[i+1] = my_PDF2CDF_hist[i] + normalize_hist[i+1]

gray_level = 255
denormalize_hist = np.zeros(256, np.float)
def my_denormalize(my_PDF2CDF_hist, gray_level) : # CDF * grey_level
    for i in range(256):
        denormalize_hist[i] = my_PDF2CDF_hist[i] * gray_level

floor = np.zeros(256, np.int)
hist_equal = np.zeros(256, np.int)
def my_calcHist_equalization(denormalized, hist) :
    for i in range(256): # 버림연산을 이용해 정수로 만듬
        floor[i] = denormalize_hist[i].astype(int) # output gray level
    for j in range(256): # 히스토그램 평활화
        for m in range(256) :
         if j == floor[m] :
            hist_equal[floor[m]] = hist[m] + hist_equal[floor[m]]

dst = np.zeros((h, w), np.int)
def my_equal_img(src, output_grey_level) : # 이미지에 평활화 적용
    for i in range(480):
        for j in range(512):
            dst[i,j] = output_grey_level[src[i,j]]

dst = dst.astype(np.uint8)

my_calcHist(src)
my_normalize_hist(hist, h * w)
my_PDF2CDF(normalize_hist)
my_denormalize(my_PDF2CDF_hist,gray_level)
my_calcHist_equalization(denormalize_hist,hist)
my_equal_img(src, floor)

map = np.zeros(256, np.int)
for i in range (256) :
    map[i] = floor[i]

plt.figure(figsize = (8,5))
cv2.imshow('equalization before image', src)
binX = np.arange(len(hist))
plt.title('my histogram')
plt.bar(binX, hist, width = 0.5, color = 'g')
plt.show()

plt.figure(figsize = (8,5))
cv2.imshow('equalization after image', dst)
binX = np.arange(len(hist_equal))
plt.title('my histogram equalization')
plt.bar(binX, hist_equal, width = 0.5, color = 'g')
plt.show()

binX = np.arange(len(map))
plt.title('mapping function')
plt.plot(binX, map)
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()