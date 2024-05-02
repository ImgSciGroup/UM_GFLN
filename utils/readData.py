import imageio
def readData(str,tinydict):
    DATASET = str
    BMP=tinydict
    img_t1 = imageio.v2.imread('./data/' + DATASET + '/T'+BMP[0]+'.' + BMP[2])  # .astype(np.float32)
    img_t2 = imageio.v2.imread('./data/' + DATASET + '/T'+BMP[1]+'.' + BMP[2])  # .astype(np.float32)
    ground_truth_changed = imageio.v2.imread('./data/' + DATASET + '/gt.' + BMP[2])
    return img_t1, img_t2, ground_truth_changed
