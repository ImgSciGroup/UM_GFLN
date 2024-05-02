import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # denom = torch.sum(adj, dim=1, keepdim=True) + 1
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class UFLN(nn.Module):
    def __init__(self, nfeat, nhid, feature, dropout):
        super(UFLN, self).__init__()
        x = feature
        x1 = x + 4
        x2 = x1 + 4
        self.gc1 = GraphConvolution(nfeat, x)
        self.gc2 = GraphConvolution(nfeat, x1)
        self.gc3 = GraphConvolution(nfeat, x2)
        self.model = nn.Sequential(
            nn.Linear(x * 2, x * 2 + 4),
            nn.LeakyReLU(inplace=True),
        )
        self.gc4 = GraphConvolution(x + x1 + x2, x * 2 + 4)  # 20
        self.gc5 = GraphConvolution(x + x1 + x2, x * 2)
        self.avg1 = torch.nn.AdaptiveAvgPool1d(1)
        self.avg2 = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, x, adj1, y, adj2):
        # T1
        #low-level feature learning
        x_fir = torch.sigmoid(self.gc1(x, adj1))
        x_sec = torch.sigmoid(self.gc2(x, adj1))
        x_thi = torch.sigmoid(self.gc3(x, adj1))
        x_f1 = torch.cat([x_fir, x_sec], 1)
        x_f2 = self.avg1(x_sec) * x_thi
        x_low_result = torch.cat([x_f1, x_f2], 1)
        # high-level feature learning
        x_fou = self.gc4(x_low_result, adj1)
        x_fiv = self.gc5(x_low_result, adj1)
        x_mlp = self.model(x_fiv)
        x_f3 = (x_mlp + x_fou) / 2
        x_low = self.avg1(x_low_result) * x_low_result + x_low_result
        x_final = torch.cat([x_low, x_f3], 1)

        # T2
        # low-level feature learning
        y_fir = torch.sigmoid(self.gc1(y, adj2))
        y_sec = torch.sigmoid(self.gc2(y, adj2))
        y_thi = torch.sigmoid(self.gc3(y, adj2))
        y_f1 = torch.cat([y_fir, y_sec], 1)
        y_f2 = self.avg1(y_sec) * y_thi
        y_low_result = torch.cat([y_f1, y_f2], 1)
        # high-level feature learning
        y_fou = self.gc4(y_low_result, adj2)
        y_fiv = self.gc5(y_low_result, adj2)
        y_mlp = self.model(y_fiv)
        y_f3 = (y_mlp + y_fou) / 2
        y_low = self.avg1(y_low_result) * y_low_result + y_low_result
        y_final = torch.cat([y_low, y_f3], 1)
        return x_low_result, y_low_result, x_final, y_final, x_fiv, x_mlp, y_fiv, y_mlp
