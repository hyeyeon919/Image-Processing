import cv2
import numpy as np

def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h+2*p_h, w+2*p_w),dtype=float)
    pad_img[p_h:p_h+h, p_w:p_w+w] = src #p_h에서부터 p_h+h전까지, p_w에서부터 p_w+w전까지 scr넣기

    if pad_type == 'repetition':
        print('repetition padding')
        #########################################################
        # TODO                                                  #
        # repetition padding 완성                                #
        #########################################################
        #up
        for i in range(p_h): #0~p_h-1
            pad_img[i, p_w:w+p_w] = src[0, 0:w]
        #down
        for i in range(p_h): #0~p_h-1
            pad_img[p_h+h+i, p_w:w+p_w] = src[h-1, 0:w]
        #left
        for i in range(p_w): #0~p_w-1
            pad_img[p_h:p_h+h, i] = src[0:h, 0]
            pad_img[0:p_h+1, i] = pad_img[0:p_h+1, p_w]
            pad_img[p_h+h-1:2*p_h+h, i] = pad_img[p_h+h-1:2*p_h+h, p_w]
        #right
        for i in range(p_w): #0~p_w-1
            pad_img[p_h:p_h+h, i+p_w+w] = src[0:h, w-1]
            pad_img[0:p_h+1, i+p_w+w] = pad_img[0:p_h+1, p_w+w-1]
            pad_img[p_h+h-1:2*p_h+h, i+p_w+w] = pad_img[p_h+h-1:2*p_h+h, p_w+w-1]

    else:
        print('zero padding')

    return pad_img

def my_filtering(src, ftype, fshape, pad_type='no'):
    global mask
    (h, w) = src.shape #흑백이미지
    src_pad = my_padding(src, (fshape[0]//2, fshape[1]//2), pad_type)
    dst = np.zeros((h, w),dtype=float)
    #ftype = mask종류
    #fshape = mask크기 - tuple(row_size, col_size)
    #pad_type = padding타입 : zero 또는 repetition

    if ftype == 'average':
        print('average filtering')
        ###################################################
        # TODO                                            #
        # mask 완성                                        #
        # 꼭 한줄로 완성할 필요 없음                           #
        ###################################################
        mask_size = fshape[0]*fshape[1]
        mask = np.full(fshape, 1/mask_size)

        #mask 확인
        #print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        ##################################################
        # TODO                                           #
        # mask 완성                                       #
        # 꼭 한줄로 완성할 필요 없음                          #
        ##################################################
        temp_mask = np.zeros(fshape,dtype=float)
        temp_mask[(fshape[0])//2, (fshape[1])//2] = 2
        mask_size = fshape[0]*fshape[1]
        mask = temp_mask - np.full(fshape,1/mask_size,dtype=float)

        #mask 확인
        print(mask)

    #########################################################
    # TODO                                                  #
    # dst 완성                                               #
    # dst : filtering 결과 image                             #
    # 꼭 한줄로 완성할 필요 없음                                 #
    #########################################################
    for row in range(h):
       for col in range(w):
           new = np.zeros(fshape,dtype=float)
           temps = np.zeros(fshape,dtype=float)
           sum = np.zeros(fshape,dtype=float)
           temps = src_pad[row : row+fshape[0], col : col+fshape[1]]
           new = mask * temps
           sum = np.sum(new, dtype=float)
           if sum > 255 :
               sum = 255
           elif sum < 0 :
               sum = 0
           dst[row, col] = sum

    dst = (dst+0.5).astype(np.uint8)

    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # repetition padding test
    rep_test = my_padding(src, (0,0), 'repetition')

    # 3x3 filter
    dst_average = my_filtering(src, 'average', (3,3), 'zero')
    dst_sharpening = my_filtering(src, 'sharpening', (3,3), 'zero')

    #원하는 크기로 설정
    #dst_average = my_filtering(src, 'average', (7,3))
    #dst_sharpening = my_filtering(src, 'sharpening', (5,6))

    # 11x13 filter
    #dst_average = my_filtering(src, 'average', (11,13))
    #dst_sharpening = my_filtering(src, 'sharpening', (11,13))

    cv2.imshow('original', src)
    cv2.imshow('average filter', dst_average)
    cv2.imshow('sharpening filter', dst_sharpening)
    cv2.imshow('repetition padding test', rep_test.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
