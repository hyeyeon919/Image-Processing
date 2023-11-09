import numpy as np

def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w), dtype=float)
    pad_img[p_h:p_h + h, p_w:p_w + w] = src  # p_h에서부터 p_h+h전까지, p_w에서부터 p_w+w전까지 scr넣기

    if pad_type == 'repetition':
        print('repetition padding')
        #########################################################
        # TODO                                                  #
        # repetition padding 완성                                #
        #########################################################
        # up
        pad_img[:p_h,p_w : w + p_w] = src[0, :]
        # down
        pad_img[p_h + h:, p_w:w + p_w] = src[h - 1, :]
        # left
        pad_img[:,:p_w] = pad_img[:,p_w:p_w+1]
        # right
        pad_img[:,p_w+w:] = pad_img[:,p_w+w-1:p_w+w]

    return pad_img
