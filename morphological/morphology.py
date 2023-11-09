import cv2
import numpy as np

def erosion(B, S):
    ###############################################
    # TODO                                        #
    # erosion 함수 완성                           #
    ###############################################
    (h, w) = B.shape
    (s_h, s_w) = S.shape
    temp_B = np.copy(B)
    for i in range (h) :
        for j in range (w) :
            if(B[i,j] == 1) :
                a = i - s_h // 2
                b = j - s_w // 2
                c = i + s_h // 2 + 1
                d = j + s_w // 2 + 1
                temp_1 = B[a:c, b:d]
                if(np.array_equal(temp_1, S) == False) :
                    temp_B[i,j] = 0
    dst = temp_B
    return dst

def dilation(B, S):
    ###############################################
    # TODO                                        #
    # dilation 함수 완성                            #
    ###############################################
    (h, w) = B.shape
    (s_h, s_w) = S.shape
    temp_B = np.copy(B)
    for i in range(h):
        for j in range(w):
            if (B[i, j] == 1):
                a = i - s_h // 2
                b = j - s_w // 2
                c = i + s_h // 2 + 1
                d = j + s_w // 2 + 1
                temp_1 = B[a:c, b:d]
                if(temp_1.shape == S.shape) :
                    temp_B[a:c, b:d] = S
    dst = temp_B
    return dst

def opening(B, S):
    ###############################################
    # TODO                                        #
    # opening 함수 완성                            #
    ###############################################
    dst = dilation(erosion(B,S),S)
    return dst

def closing(B, S):
    ###############################################
    # TODO                                        #
    # closing 함수 완성                            #
    ###############################################
    dst = erosion(dilation(B,S),S)
    return dst


if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])


    cv2.imwrite('morphology_B.png', (B*255).astype(np.uint8))

    img_dilation = dilation(B, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('morphology_dilation.png', img_dilation)

    img_erosion = erosion(B, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('morphology_erosion.png', img_erosion)


    img_opening = opening(B, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)


    img_closing = closing(B, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)

    img_B = cv2.resize((B*255).astype(np.uint8), dsize=(400, 350), interpolation=cv2.INTER_AREA)
    img_dilation = cv2.resize(img_dilation, dsize=(400, 350), interpolation=cv2.INTER_AREA)
    img_erosion = cv2.resize(img_erosion, dsize=(400, 350), interpolation=cv2.INTER_AREA)
    img_opening = cv2.resize(img_opening, dsize=(400, 350), interpolation=cv2.INTER_AREA)
    img_closing = cv2.resize(img_closing, dsize=(400, 350), interpolation=cv2.INTER_AREA)

    cv2.imshow('morphology_B.png', img_B)
    cv2.imshow('morphology_dilation.png', img_dilation)
    cv2.imshow('morphology_erosion.png', img_erosion)
    cv2.imshow('morphology_opening.png', img_opening)
    cv2.imshow('morphology_closing.png', img_closing)
    cv2.waitKey()
    cv2.destroyAllWindows()


