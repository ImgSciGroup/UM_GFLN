import numpy as np
import cv2 as cv

def norm_data(data):
    data_height, data_width = data.shape
    data = np.reshape(data, (data_height * data_width))  # (channel, height * width)
    max = np.max(data, axis=0, keepdims=True)  # (channel, 1)
    min = np.min(data, axis=0, keepdims=True)  # (channel, 1)
    diff_value = max - min
    nm_data = (data - min) / diff_value
    nm_data = np.reshape(nm_data, (data_height, data_width))
    return nm_data
def getResult(objects,obj_nums,diff_set,height, width):
    CMI = np.zeros((height, width))
    for i in range(1, obj_nums):
        CMI[objects == i] = diff_set[i-1]
    CMI=norm_data(CMI)*255
    CMI=CMI.astype(np.uint8)
    threshold, a_img = cv.threshold(CMI, 0, 255, cv.THRESH_OTSU)
    bcm = np.zeros((height, width)).astype(np.uint8)
    bcm[CMI > threshold] = 255
    bcm[CMI <= threshold] = 0
    return CMI,bcm
def Acc(GT_C, GT_UC, BCM):
    C_IDX = (GT_C == 255)
    UC_IDX = (GT_UC == 255)
    cc = (BCM[C_IDX] == 255).sum()
    uu = (BCM[UC_IDX] == 0).sum()
    cu = C_IDX.sum() - cc
    uc = UC_IDX.sum() - uu
    conf_mat = np.array([[cc, cu], [uc, uu]])
    pre = cc / (cc + uc)
    rec = cc / (cc + cu)
    f1 = 2 * pre * rec / (pre + rec)
    over_acc = (conf_mat.diagonal().sum()) / (conf_mat.sum())
    # pe = np.array(0, np.int64)
    pe = ((cc + cu) / conf_mat.sum() * (cc + uc) + (uu + uc) / conf_mat.sum() * (
            uu + cu)) / conf_mat.sum()
    kappa_co = (over_acc - pe) / (1 - pe)
    return conf_mat, over_acc, f1, kappa_co


