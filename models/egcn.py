import torch
import torch.nn as nn
import numpy as np
from networkx.algorithms import tree
from .utils import extractor, padder, averager

__all__ = ['egcn']

class egcn(nn.Module):
    def __init__(self, C_in, C_out):
        super(egcn, self).__init__()
        self.squeeze = nn.Sequential(nn.Conv1d(in_channels=2*C_in, out_channels=C_out, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm1d(C_out),
                                   nn.ReLU(True))
        self.softmax_1 = nn.Softmax(dim=-1)
        self.softmax_2 = nn.Softmax(dim=-2)
        self.trans = nn.Sequential(nn.Conv1d(in_channels=C_in, out_channels=C_out, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm1d(C_out),
                                   nn.ReLU(True))
        self.fusion = nn.Sequential(nn.Conv2d(in_channels=C_out, out_channels=C_out//2, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(C_out//2),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channels=C_out//2, out_channels=C_out//2, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(C_out//2),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channels=C_out//2, out_channels=C_out, kernel_size=1, stride=1,padding=0),
                                    )
        self.C_out = C_out

    def forward(self, x):
        B,C,H,W = x.size()
        size = H
        assert H == W
        x = x.view(B,C,-1)
        I = x
        x = torch.cat([self.softmax_1(x), self.softmax_2(x)],dim=1)
        upper = x[:, :, 0:-size] # 0 to (HW - size)
        upper = self.squeeze(upper)
        mask_upper = upper
        upper = torch.cat([torch.zeros([B,self.C_out,size],device='cuda',requires_grad=True), upper], dim=2)

        mid_upper = extractor(x[:, :, 0:-1], size)
        mid_upper = self.squeeze(mid_upper)
        mask_mid_upper = mid_upper
        mid_upper = padder(mid_upper, size)
        mid_upper = torch.cat([torch.zeros([B,self.C_out,1],device='cuda',requires_grad=True), mid_upper], dim=2)

        lower = x[:, :, size:] # size to HW
        lower = self.squeeze(lower)
        mask_lower = lower
        lower = torch.cat([lower, torch.zeros([B,self.C_out,size],device='cuda',requires_grad=True)], dim=2)

        mid_lower = extractor(x[:, :, 1:], size)
        mid_lower = self.squeeze(mid_lower)
        mask_mid_lower = mid_lower
        mid_lower = padder(mid_lower, size)
        mid_lower = torch.cat([mid_lower, torch.zeros([B,self.C_out,1],device='cuda',requires_grad=True)], dim=2)

        gnn = upper + mid_upper + lower + mid_lower + self.trans(I)
        gnn = self.fusion(gnn.view(B,self.C_out,H,W))

        return gnn, [mask_upper, mask_mid_upper, mask_lower, mask_mid_lower]

if __name__ == '__main__':
    mu, sigma = 0, 0.1  # mean and standard deviation
    b, c, h, w = 64, 64, 16, 16
    image_x = np.random.normal(mu, sigma, b * c * h * w).reshape((b, c, h, w))
    x = torch.tensor(image_x).float()
    net = egcn(c)
    gnn, mask = net(x)
    import time
    start = time.time()
    for i in range(1):
        np.shape(mask2dist(mask, h)[0])
    end = time.time()
    print((end-start)/1)