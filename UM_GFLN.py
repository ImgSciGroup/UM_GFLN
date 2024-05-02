import argparse
import time
import imageio
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from utils.Dataload import dataload
from utils.Result import getResult, Acc
from utils.graph_func import find_sim_node, graph_construct
import yaml
from utils.loss import cal_loss
from model.GFLN import UFLN
def train_test(args):
    with open('./config.yaml', 'r', encoding='utf-8') as f:
        conf = yaml.load(f.read(), Loader=yaml.FullLoader)
    k = conf['k']
    weight = conf['weight']
    feat = conf['feat']
    conf=conf[args.dataset]
    EPOCH=conf['EPOCH']
    im1, im2, GT_C , objects=dataload(args,conf)
    GT_UC = 255 - GT_C
    DATASET=args.dataset
    obj_nums = np.max(objects) + 1
    height, width, channel_t1 = im1.shape
    nodelist_x = []
    nodelist_y = []
    for index in range(1,obj_nums):
        obj_index = objects == index
        x_node=im1[obj_index]
        y_node=im2[obj_index]
        x_node=find_sim_node(x_node,k)
        y_node=find_sim_node(y_node,k)
        nodelist_x.append(x_node)
        nodelist_y.append(y_node)
    adjlist_x = graph_construct(nodelist_x)
    adjlist_y = graph_construct(nodelist_y)
    print('----图构造完成----')
    UFLN_model = UFLN(nfeat=3, nhid=10, feature=feat, dropout=0.5)
    optimizer = optim.AdamW(UFLN_model.parameters(), lr=1e-4, weight_decay=1e-6)
    UFLN_model.cuda()
    UFLN_model.train()
    for _epoch in tqdm(range(EPOCH)):
        for i in range(obj_nums-1):
            x_node = nodelist_x[i]
            x_adj, n_x_adj = adjlist_x[i]
            x_node = torch.from_numpy(x_node).cuda().float()
            x_adj = torch.from_numpy(x_adj).cuda().float()
            n_x_adj = torch.from_numpy(n_x_adj).cuda().float()
            y_node = nodelist_y[i]
            y_adj, n_y_adj = adjlist_y[i]
            y_node = torch.from_numpy(y_node).cuda().float()
            y_adj = torch.from_numpy(y_adj).cuda().float()
            n_y_adj = torch.from_numpy(n_y_adj).cuda().float()
            x_low_result, y_low_result, x_final, y_final, x_fiv, x_mlp, y_fiv, y_mlp=\
                UFLN_model(x_node, n_x_adj, y_node, n_y_adj)
            loss_co ,loss_x_final,loss_y_final,loss_x_low,loss_y_low,loss_x_mlp,loss_y_mlp=\
                cal_loss( x_low_result, y_low_result, x_final, y_final, x_fiv, x_mlp, y_fiv, y_mlp,x_adj,y_adj)
            total_loss = weight * loss_co + loss_x_final+loss_y_final+loss_x_low+loss_y_low+loss_x_mlp+loss_y_mlp
            total_loss.backward()
            optimizer.step()
    torch.save(UFLN_model.state_dict(), './model_file/' +str(
            DATASET) +'__'+ str(time.time())+'__' +str(k)+'__'+str(weight)+'__'+str(feat)+ '.pth')
    print('----训练完成----')
    UFLN_model.eval()
    cmi_set1 = []
    diff_re = []
    for i in range(obj_nums-1):
        x_node = nodelist_x[i]
        y_node = nodelist_y[i]
        x_adj, n_x_adj = adjlist_x[i]
        y_adj, n_y_adj = adjlist_y[i]
        x_node = torch.from_numpy(x_node).cuda().float()
        y_node = torch.from_numpy(y_node).cuda().float()
        n_x_adj = torch.from_numpy(n_x_adj).cuda().float()
        n_y_adj = torch.from_numpy(n_y_adj).cuda().float()
        x_low_result, y_low_result, x_final, y_final, x_fiv, x_mlp, y_fiv, y_mlp= UFLN_model(x_node, n_x_adj, y_node, n_y_adj)
        diff1 = torch.mean(torch.abs(x_final - y_final))
        cmi_set1.append(diff1.data.cpu().numpy())
    diff_re.append(cmi_set1)
    for i in range(len(diff_re)):
        CMI, bcm = getResult(objects, obj_nums, diff_re[i], height, width)
        mat, oa, f1, kappa_co = Acc(GT_C, GT_UC, bcm)
        imageio.imsave('./result/' + str(oa) + '_' + str(EPOCH) + '_' + str(time.time()) + '----' + str(
            DATASET) + '_k=' + str(k)+'weight='+str(weight)+'feature'+str(feat)+ '.png', bcm)
        CMI = 255 * (CMI - np.min(CMI)) / (np.max(CMI) - np.min(CMI))
        imageio.imsave('./result/' + str(oa) + '_' + str(EPOCH) + '_' + str(time.time()) + '----' + str(
            DATASET) + '_k=' + str(k)+'weight='+str(weight)+'feature'+str(feat) + '_DI.png', CMI.astype(np.uint8))
        print('OA'+'----'+str(oa))
        print('F1'+'----'+str(f1))
        print('Ka'+'----'+str(kappa_co))
    print('----测试完成----')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset', type=str, default='UK', help='Training dataset of UM_GFLN') #'TE' 'UK' 'UK2' 'Lss' 'SG' 'FR'
    args = parser.parse_args()
    train_test(args)
