import numpy as np
from numpy import argsort
from sklearn.metrics.pairwise import pairwise_distances

def pr_distance(node):
    euc_dis = pairwise_distances(node)
    a = 1 - (np.sqrt(np.sum(euc_dis)) - np.mean(euc_dis)) / (np.sqrt(np.sum(euc_dis)) + np.mean(euc_dis))
    gaus_dis = np.exp(- euc_dis * euc_dis / (a))
    return gaus_dis

def graph_construct(objects):
    adjlist = []
    obj_nums = len(objects)
    for i in range(0, obj_nums):
        sub_object = objects[i]
        adj_mat = pr_distance(sub_object)
        norm_adj_mat= normalize_adj(adj_mat)
        adjlist.append([adj_mat, norm_adj_mat])
    return adjlist
def find_sim_node(node,k):
    size = node.shape[0]
    ####计算欧式距离
    diff = pairwise_distances(node)
    node_1=[]
    for i in range(size):
        a=diff[i,:]
        sortedDistIndex = argsort(a)
        sortedDistIndex1=sortedDistIndex[::k]
        list = np.array(sortedDistIndex1)
        node_re=node[list]
        node_1.append(node_re)
    node=np.array(node_1)
    node_re=np.mean(node,axis=0)#0
    return  node_re
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    d_inv_sqrt = np.power(np.array(adj.sum(1)) , -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    return adj.dot(np.diag(d_inv_sqrt)).transpose().dot(np.diag(d_inv_sqrt))