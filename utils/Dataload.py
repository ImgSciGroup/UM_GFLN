from utils.preprocess import preprocess_img
from utils.readData import readData
from skimage.segmentation import slic

def dataload(args,conf):
    #加载数据
    str=args.dataset
    seg = conf['SEGMENT']

    # seg=args.seg[str]
    dict_loadData=conf['DATANAME']
    img_t1,img_t2,GC=readData(str,dict_loadData)
    #超像素分割
    objects = slic(img_t2, n_segments=seg[0], compactness=seg[1])
    #归一化
    norm_type= conf['GUIYI']

    img_t1=preprocess_img(img_t1,'opt',norm_type)
    img_t2=preprocess_img(img_t2,'opt',norm_type)
    return img_t1,img_t2,GC,objects



