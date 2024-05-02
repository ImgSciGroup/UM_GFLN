import torch
import torch.nn.functional as F
def cal_loss( x_low_result, y_low_result, x_final, y_final, x_fiv, x_mlp, y_fiv, y_mlp,x_adj,y_adj):
    re_x_low = torch.matmul(x_low_result, x_low_result.T)
    re_y_low = torch.matmul(y_low_result, y_low_result.T)
    re_x_final = torch.matmul(x_final, x_final.T)
    re_y_final = torch.matmul(y_final, y_final.T)
    re_x_fiv = torch.matmul(x_fiv, x_fiv.T)
    re_x_mlp = torch.matmul(x_mlp, x_mlp.T)
    re_y_fiv = torch.matmul(y_fiv, y_fiv.T)
    re_y_mlp = torch.matmul(y_mlp, y_mlp.T)
    NUM = y_adj.size()[0]
    loss_co = F.l1_loss(input=x_final, target=y_final)
    loss_x_low = F.mse_loss(input=re_x_low, target=x_adj) / NUM
    loss_y_low = F.mse_loss(input=re_y_low, target=y_adj) / NUM
    loss_x_final = F.mse_loss(input=re_x_final, target=x_adj) / NUM
    loss_y_final = F.mse_loss(input=re_y_final, target=y_adj) / NUM
    loss_x_mlp = F.mse_loss(input=re_x_fiv, target=re_x_mlp) / NUM
    loss_y_mlp = F.mse_loss(input=re_y_fiv, target=re_y_mlp) / NUM
    return loss_co ,loss_x_final,loss_y_final,loss_x_low,loss_y_low,loss_x_mlp,loss_y_mlp