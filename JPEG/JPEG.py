import numpy as np
import cv2
import time

def Quantization_Luminance():
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance

def img2block(src, n=8):
    blocks = []
    (h,w) = src.shape
    for i in range (h//n) :
        for j in range (w//n) :
            blocks.append(src[i * n:(i + 1) * n, j * n:(j + 1) * n])
    return np.array(blocks)

def C(w, n = 8):
    if w == 0:
        return (1/n)**0.5
    else:
        return (2/n)**0.5

def DCT(block, n=8):
    dst = np.zeros(block.shape)
    v, u = dst.shape

    y, x = np.mgrid[0:v, 0:u]
    mask = np.zeros((n*n,n*n))

    for v_ in range(v):
        for u_ in range(u):
            temp = block[y, x] * np.cos(((2 * x + 1) * u_ * np.pi) / (2 * n)) * \
                   np.cos(((2 * y + 1) * v_ * np.pi) / (2 * n))

            dst[v_, u_] = C(v_, n) * C(u_, n) * np.sum(temp)
    return np.round(dst)

def my_zigzag_scanning(arr, mode='encoding', block_size=8):
    if mode == 'encoding' :
        z_search = []
        index_x = 0
        index_y = 0
        z_search.append(arr[0][0])
        while index_x!=block_size and index_y!=block_size:
            if index_y == block_size-1 and index_x == block_size-1 :
                break
            elif index_y == block_size-1 and index_x == 0:
                index_x=index_x+1
                z_search.append(arr[index_y][index_x])
                while index_x < block_size-1 :
                    index_x = index_x + 1
                    index_y = index_y - 1
                    z_search.append(arr[index_y][index_x])
            elif index_y == 0:
                index_x=index_x+1
                z_search.append(arr[index_y][index_x])
                while index_x != 0 :
                    index_x = index_x - 1
                    index_y = 1 + index_y
                    z_search.append(arr[index_y][index_x])
            elif index_x == 0 :
                index_y = 1 + index_y
                z_search.append(arr[index_y][index_x])
                while index_y > 0 :
                    index_x = 1 + index_x
                    index_y = index_y -1
                    z_search.append(arr[index_y][index_x])
            elif index_x == block_size - 1:
                index_y = 1 + index_y
                z_search.append(arr[index_y][index_x])
                while index_y < block_size - 1 :
                    index_x = index_x - 1
                    index_y = 1 + index_y
                    z_search.append(arr[index_y][index_x])
            elif index_y == block_size - 1:
                index_x = 1 + index_x
                z_search.append(arr[index_y][index_x])
                if index_x < block_size-1 or index_y < block_size-1 :
                    while index_x < block_size - 1 :
                        index_x = 1 + index_x
                        index_y = index_y -1
                        z_search.append(arr[index_y][index_x])
        for i in range (len(z_search)) :
            test_arr2 = z_search[i: i + 16]
            if (test_arr2) == [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] :
                z_search = z_search[0:i]
                z_search.append('EOB')
    else :
        new_arr = []
        arr = np.array(arr)
        i = 0
        while i<len(arr) :
            if((arr[i])!='EOB') :
                new_arr.append(arr[i])
            i = i+1
        z_search = np.zeros((block_size,block_size))
        index_x = 0
        index_y = 0
        z_search[0][0] = arr[0]
        cnt = 0
        while cnt < len(new_arr) :#new_arr가 끝날 때까지 시행 EOB전까지 시행
            if index_y == block_size - 1 and index_x == 0:
                index_x = index_x + 1
                cnt = cnt+1
                if len(new_arr) == cnt :
                    break
                z_search[index_y][index_x] = new_arr[cnt]
                while index_x < block_size - 1:
                    index_x = index_x + 1
                    index_y = index_y - 1
                    cnt = cnt + 1
                    if len(new_arr) == cnt:
                        break
                    z_search[index_y][index_x] = new_arr[cnt]
            elif index_y == 0:
                index_x = index_x + 1
                cnt = cnt + 1
                if len(new_arr) == cnt:
                    break
                z_search[index_y][index_x] = new_arr[cnt]
                while index_x != 0:
                    index_x = index_x - 1
                    index_y = 1 + index_y
                    cnt = cnt + 1
                    if len(new_arr) == cnt:
                        break
                    z_search[index_y][index_x] = new_arr[cnt]
            elif index_x == 0:
                index_y = 1 + index_y
                cnt = cnt + 1
                if len(new_arr) == cnt:
                    break
                z_search[index_y][index_x] = new_arr[cnt]
                while index_y != 0:
                    index_x = 1 + index_x
                    index_y = index_y - 1
                    cnt = cnt + 1
                    if len(new_arr) == cnt:
                        break
                    z_search[index_y][index_x] = new_arr[cnt]

            elif index_x == block_size - 1:
                index_y = 1 + index_y
                cnt = cnt + 1
                if len(new_arr) == cnt:
                    break
                z_search[index_y][index_x] = new_arr[cnt]
                while index_y != block_size - 1:
                    index_x = index_x - 1
                    index_y = 1 + index_y
                    cnt = cnt + 1
                    if len(new_arr) == cnt:
                        break
                    z_search[index_y][index_x] = new_arr[cnt]
            elif index_y == block_size - 1:
                index_x = 1 + index_x
                cnt = cnt + 1
                if len(new_arr) == cnt:
                    break
                z_search[index_y][index_x] = new_arr[cnt]
                if index_x < block_size - 1 or index_y < block_size - 1:
                    while index_x != block_size - 1:
                        index_x = 1 + index_x
                        index_y = index_y - 1
                        cnt = cnt + 1
                        if len(new_arr) == cnt:
                            break
                        z_search[index_y][index_x] = new_arr[cnt]
    return z_search


def DCT_inv(block, n = 8):
    dst = np.zeros((n,n))
    v, u = dst.shape
    y, x = np.mgrid[0:v, 0:u]
    mask = np.zeros((n*n,n*n))
    C_xtemp = np.full(n,(2/n)**0.5)
    C_xtemp[0] = (1/n)**0.5
    C_ytemp = np.full(n,(2/n)**0.5)
    C_ytemp[0] = (1/n)**0.5
    for v_ in range(v):#y
        for u_ in range(u):#x
            temp = block[y, x] * C_xtemp[x] * C_ytemp[y] * np.cos(((2 * u_ + 1) * x * np.pi) / (2 * n)) * \
                   np.cos(((2 * v_ + 1) * y * np.pi) / (2 * n))
            dst[v_, u_] = np.sum(temp)
    return np.round(dst)

def block2img(blocks, src_shape, n = 8):
    (dst_h, dst_w) = src_shape # 512,512
    dst = np.zeros(src_shape)
    cnt = 0
    for i in range(dst_h//(n)) : # 아래로 추가 8
        for j in range(dst_w//(n)) : # 오른쪽으로 추가 8
            if cnt < len(blocks) : #블럭의 수 만큼 시행
                dst[(i) * n:(i + 1) * n, (j) * n:(j + 1) * n] = blocks[cnt]
                #block.shape = 8,8
                cnt = cnt+1
                #blocks의 인덱스 가산
    return dst

def Encoding(src, n=8):
    #################################################################################################
    # TODO                                                                                          #
    # Encoding 완성                                                                                  #
    # Encoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Encoding>')
    # img -> blocks
    blocks = img2block(src, n=n)

    #subtract 128
    blocks -= 128
    #DCT
    blocks_dct = []
    for block in blocks:
        blocks_dct.append(DCT(block, n=n))
    blocks_dct = np.array(blocks_dct)

    #Quantization + thresholding
    Q = Quantization_Luminance()
    QnT = np.round(blocks_dct / Q)
    # zigzag scanning

    zz = []
    for i in range(len(QnT)): # len(QnT) = 64
        zz.append(my_zigzag_scanning(QnT[i]))
    return zz, src.shape


def Decoding(zigzag, src_shape, n=8):
    #################################################################################################
    # TODO                                                                                          #
    # Decoding 완성                                                                                  #
    # Decoding 함수를 참고용으로 첨부하긴 했는데 수정해서 사용하실 분은 수정하셔도 전혀 상관 없습니다.              #
    #################################################################################################
    print('<start Decoding>')

    # zigzag scanning
    blocks = []
    for i in range(len(zigzag)):
        blocks.append(my_zigzag_scanning(zigzag[i], mode='decoding', block_size=n))
    blocks = np.array(blocks)

    # Denormalizing
    Q = Quantization_Luminance()
    blocks = blocks * Q

    # inverse DCT
    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))
    blocks_idct = np.array(blocks_idct)
    # add 128
    blocks_idct += 128

    # block -> img
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)
    # print(dst.shape)
    return dst


def main():
    start = time.time()
    # src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('original img', src)

    # comp, src_shape = Encoding(src, n=8)
    # 과제의 comp.npy, src_shape.npy를 복구할 때 아래 코드 사용하기(위의 2줄은 주석처리하고, 아래 2줄은 주석 풀기)
    comp = np.load('comp.npy', allow_pickle=True)
    src_shape = np.load('src_shape.npy')
    recover_img = Decoding(comp, src_shape, n=8)
    total_time = time.time() - start
    print('time : ', total_time)
    if total_time > 45:
        print('감점 예정입니다.')

    # comp.npy 출력할떄 오버플로우, 언더플로우 처리
    # recover_img = np.clip(recover_img,0,255)
    recover_img = recover_img.astype(np.uint8)

    cv2.imshow('recover img', recover_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
