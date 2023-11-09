import numpy as np
import cv2

def forward(src, M, fit=False):
    (h,w) = src.shape
    print('< forward >')
    print('M')
    print(M)

    if fit == False :
        dst = np.zeros(src.shape)
        dst_cnt = np.zeros(src.shape)
        for row in range (h) :
            for col in range (w) :
                p = np.array([
                    [col],
                    [row],
                    [1]
                ])
                p_dst = M.dot(p)
                dst_col = p_dst[0][0]
                dst_row = p_dst[1][0]
                dst_col_r = int(np.ceil(dst_col))
                dst_col_l = int(dst_col)
                dst_row_b = int(np.ceil(dst_row))
                dst_row_t = int(dst_row)

                if dst_row>=0 and dst_col>=0 and dst_row_b < h and dst_col_r < w :
                    # print((dst_row, dst_col))
                    dst[dst_row_t, dst_col_l] += src[row, col]
                    dst_cnt[dst_row_t, dst_col_l] += 1

                    if dst_col_r != dst_col_l :
                        dst[dst_row_t, dst_col_r] += src[row,col]
                        dst_cnt[dst_row_t, dst_col_r] += 1

                    if dst_row_b != dst_row_t :
                        dst[dst_row_b, dst_col_l] += src[row,col]
                        dst_cnt[dst_row_b, dst_col_l] += 1

                    if dst_col_r != dst_col_l and dst_row_b != dst_row_t :
                        dst[dst_row_b, dst_col_r] += src[row, col]
                        dst_cnt[dst_row_b, dst_col_r] += 1
    else :
        p = np.array([ #(0,0)
            [0],
            [0],
            [1]
        ])
        p_dst = M.dot(p)
        dst_col_00 = p_dst[0][0]
        dst_row_00 = p_dst[1][0]

        p = np.array([ #(h-1,0)
            [0],
            [h-1],
            [1]
        ])
        p_dst = M.dot(p)
        dst_col_h0 = p_dst[0][0]
        dst_row_h0 = p_dst[1][0]

        p = np.array([ #(0,w-1)
            [w-1],
            [0],
            [1]
        ])
        p_dst = M.dot(p)
        dst_col_w0 = p_dst[0][0]
        dst_row_w0 = p_dst[1][0]

        p = np.array([ #(h-1,w-1)
            [w-1],
            [h-1],
            [1]
        ])
        p_dst = M.dot(p)
        dst_col_hw = p_dst[0][0]
        dst_row_hw = p_dst[1][0]

        if dst_row_w0 < dst_row_00 : # 가장 작은 x값
            min_row = dst_row_w0
        else :
            min_row = dst_row_00

        if dst_col_h0 < dst_col_00 : # 가장 작은 y값
            min_col = dst_col_h0
        else :
            min_col = dst_col_00

        if dst_col_hw < dst_col_w0 : # 가장 큰 x값
            max_col = dst_col_w0
        else :
            max_col = dst_col_hw

        if dst_row_hw < dst_row_h0 : # 가장 큰 y값
            max_row = dst_row_h0
        else :
            max_row = dst_row_hw

        new_w = int(np.ceil(max_col - min_col))
        new_h = int(np.ceil(max_row - min_row))

        dst = np.zeros((new_h, new_w))
        dst_cnt = np.zeros((new_h, new_w))

        for row in range (h) :
            for col in range (w) :
                p = np.array([
                    [col],
                    [row],
                    [1]
                ])
                p_dst = M.dot(p)
                dst_col = p_dst[0][0] + abs(min_col)
                dst_row = p_dst[1][0] + abs(min_row)
                dst_col_r = int(np.ceil(dst_col))
                dst_col_l = int(dst_col)

                dst_row_b = int(np.ceil(dst_row))
                dst_row_t = int(dst_row)
                if dst_row_b < new_h and dst_col_r < new_w:

                    dst[dst_row_t, dst_col_l] += src[row, col]
                    dst_cnt[dst_row_t, dst_col_l] += 1

                    if dst_col_r != dst_col_l :
                        dst[dst_row_t, dst_col_r] += src[row,col]
                        dst_cnt[dst_row_t, dst_col_r] += 1

                    if dst_row_b != dst_row_t :
                        dst[dst_row_b, dst_col_l] += src[row,col]
                        dst_cnt[dst_row_b, dst_col_l] += 1

                    if dst_col_r != dst_col_l and dst_row_b != dst_row_t :
                        dst[dst_row_b, dst_col_r] += src[row, col]
                        dst_cnt[dst_row_b, dst_col_r] += 1

    dst = np.round(dst / (dst_cnt + 1E-6))
    dst = dst.astype(np.uint8)
    return dst

def backward(src, M, fit=False):
    print('< backward >')
    print('M')
    print(M)

    M_inv = np.linalg.inv(M)
    print('M inv')
    print(M_inv)

    if fit == False :
        dst_tmp = np.zeros(src.shape)
        (h, w) = dst_tmp.shape
        h_src, w_src = src.shape
        for row in range (h) :
            for col in range (w) :
                p_dst = np.array([
                    [col],
                    [row],
                    [1]
                ])
                p = M_inv.dot(p_dst)
                src_col = p[0,0]
                src_row = p[1,0]

                src_col_r = int(np.ceil(src_col))
                src_col_l = int(src_col)

                src_row_b = int(np.ceil(src_row))
                src_row_t = int(src_row)
                if src_col_r >= w_src or src_row_b >= h_src :
                    continue

                if src_col >= 0 and src_col < w and src_row>=0 and src_row<h :
                    s = src_col - src_col_l
                    t = src_row - src_row_t

                    intensity = (1-s) * (1-t) * src[src_row_t, src_col_l] \
                                + s * (1-t) * src[src_row_t, src_col_r] \
                                + (1-s) * t * src[src_row_b, src_col_l] \
                                + s * t * src[src_row_b, src_col_r]
                    dst_tmp[row, col] = intensity

    else :
        (h, w) = src.shape
        h_src, w_src = src.shape
        #p_dst의 양끝 모서리 위치 구하기
        p_dst = np.array([  # (0,0)
            [0],
            [0],
            [1]
        ])
        p = M.dot(p_dst)
        src_col_00 = p[0, 0]
        src_row_00 = p[1, 0]

        p_dst = np.array([  # (h-1,0)
            [0],
            [h - 1],
            [1]
        ])
        p = M.dot(p_dst)
        src_col_h0 = p[0,0]
        src_row_h0 = p[1,0]

        p_dst = np.array([  # (0,w-1)
            [w - 1],
            [0],
            [1]
        ])
        p = M.dot(p_dst)
        src_col_w0 = p[0,0]
        src_row_w0 = p[1,0]

        p_dst = np.array([  # (h-1,w-1)
            [w - 1],
            [h - 1],
            [1]
        ])
        p = M.dot(p_dst)
        src_col_hw = p[0,0]
        src_row_hw = p[1,0]

        #모서리의 끝을 구해서 dst_tmp크기 구하기
        if src_row_w0 < src_row_00 : # 가장 작은 x값
            min_row = src_row_w0
        else :
            min_row = src_row_00

        if src_col_h0 < src_col_00 : # 가장 작은 y값
            min_col = src_col_h0
        else :
            min_col = src_col_00

        if src_col_hw < src_col_w0 : # 가장 큰 x값
            max_col = src_col_w0
        else :
            max_col = src_col_hw

        if src_row_hw < src_row_h0 : # 가장 큰 y값
            max_row = src_row_h0
        else :
            max_row = src_row_hw

        new_w = int(np.ceil(max_col - min_col))
        new_h = int(np.ceil(max_row - min_row))

        dst_tmp = np.zeros((new_h, new_w))

        #warping
        for row in range (int(new_h + abs(min_row))) :
            for col in range (int(new_w + abs(min_col))) :
                p_dst = np.array([
                    [col],
                    [row],
                    [1]
                ])
                p = M_inv.dot(p_dst)
                src_col = p[0,0]
                src_row = p[1,0]

                src_col_r = int(np.ceil(src_col))
                src_col_l = int(src_col)

                src_row_b = int(np.ceil(src_row))
                src_row_t = int(src_row)

                if src_col_r >= w_src or src_row_b >= h_src :
                    continue

                s = src_col - src_col_l
                t = src_row - src_row_t

                rows = int(row+ abs(min_row))
                cols = int(col+ abs(min_col))

                if src_row_t > 0 and src_col_l > 0:
                    intensity = (1-s) * (1-t) * src[src_row_t, src_col_l] \
                                + s * (1-t) * src[src_row_t, src_col_r] \
                                + (1-s) * t * src[src_row_b, src_col_l] \
                                + s * t * src[src_row_b, src_col_r]
                    dst_tmp[rows, cols] = intensity
    dst = dst_tmp.astype(np.uint8)
    return dst

def main():
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)

    # translation
    M_tr = np.array([
        [1, 0, -30],
        [0, 1, 50],
        [0, 0, 1]
    ])

    # scaling
    M_sc = np.array([
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 1]
    ])

    # rotation
    degree = -20
    M_ro = np.array([
        [np.cos(np.deg2rad(degree)), -np.sin(np.deg2rad(degree)), 0],
        [np.sin(np.deg2rad(degree)), np.cos(np.deg2rad(degree)), 0],
        [0, 0, 1]
    ])
    # shearing
    M_sh = np.array([
        [1, 0.2, 0],
        [0.2, 1, 0],
        [0, 0, 1]
    ])
    # rotation -> translation -> Scale -> Shear
    M = M_tr.dot(M_ro)
    M = M_sc.dot(M)
    M = M_sh.dot(M)

    # fit이 True인 경우와 False인 경우 다 해야 함.
    fit = False
    # # forward
    dst_for = forward(src, M, fit=fit)
    dst_for2 = forward(dst_for, np.linalg.inv(M), fit=fit)

    # backward
    dst_back = backward(src, M, fit=fit)
    dst_back2 = backward(dst_back, np.linalg.inv(M), fit=fit)

    cv2.imshow('original', src)
    cv2.imshow('forward', dst_for)
    cv2.imshow('forward2', dst_for2)
    cv2.imshow('backward', dst_back)
    cv2.imshow('backward2', dst_back2)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ =='__main__':
    main()
