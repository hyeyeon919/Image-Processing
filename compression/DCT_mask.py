import cv2
import numpy as np
#jpeg는 보통 block size = 8


def C(w, n = 8):
    if w == 0:
        return (1/n)**0.5
    else:
        return (2/n)**0.5


def Spatial2Frequency_mask(block, n = 8):
    dst = np.zeros(block.shape)
    v, u = dst.shape

    y, x = np.mgrid[0:u, 0:v]
    mask = np.zeros((n*n,n*n))

    for v_ in range(v):
        for u_ in range(u):
            temp = my_normalize(np.cos(((2 * x + 1) * u_ * np.pi) / (2 * n)) *
                                np.cos(((2 * y + 1) * v_ * np.pi) / (2 * n)))
            mask[v_*n:(v_+1)*n, u_*n:(u_+1)*n] = temp

    print(mask)
    return mask

def my_normalize(src):
    ##############################################################################
    # ToDo                                                                       #
    # my_normalize                                                               #
    # mask를 보기 좋게 만들기 위해 어떻게 해야 할 지 생각해서 my_normalize 함수 완성해보기   #
    ##############################################################################
    #dst => 0 ~ 255
    dst = src.copy()
    if np.max(dst) - np.min(dst):
        dst = (dst - np.min(dst)) / (np.max(dst) - np.min(dst)) * 255
    else:
        dst = dst * 255
    return dst.astype(np.uint8)

if __name__ == '__main__':
    block_size = 4
    src = np.ones((block_size, block_size))

    mask = Spatial2Frequency_mask(src, n=block_size)
    #크기가 너무 작으니 크기 키우기 (16x16) -> (320x320)
    mask = cv2.resize(mask, (320, 320), interpolation=cv2.INTER_NEAREST)
    print(mask)
    cv2.imshow('201902739 mask', my_normalize(mask))
    cv2.waitKey()
    cv2.destroyAllWindows()



