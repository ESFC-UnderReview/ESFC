import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .lzw import get_lzw_length

__all__ = ['auto_correlation', 'KDLoss', 'lzw_code']

class auto_correlation(nn.Module):
    def __init__(self, ):
        super(auto_correlation, self).__init__()

    def forward(self, image, k=3):
        B,C,H,W = image.size()
        image = image.view(B,C,-1)
        yy = torch.sum(image*image, dim=-1)
        y1 = image[:,:,:-k]
        y2 = image[:,:,k:]
        y1y2 = torch.sum(y1*y2, dim=-1)
        return torch.mean(y1y2/yy)

class lzw_code(nn.Module):
    def __init__(self, ):
        super(lzw_code, self).__init__()
        self.score = 0
    def forward(self, image, p_coord):
        B,C,H,W = image.size()
        for i in range(B):
            score = get_lzw_length((np.array(image_array[i,:,:,:]), p_coord))
            self.score += score
        self.score = self.score/B
        return self.score

class KDLoss(nn.Module): # Reference: https://github.com/irfanICMLL/structure_knowledge_distillation
    def __init__(self, T=1):
        super(KDLoss, self).__init__()
        self.T = T
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, student, teacher):
        teacher.detach()
        B, C, W, H = student.size()
        teacher_output = teacher.permute(0, 2, 3, 1).contiguous().view(-1, C)
        output = student.permute(0, 2, 3, 1).contiguous().view(-1, C)
        student_outout = self.logsoftmax(output / self.T)
        softmax_pred_T = F.softmax(teacher_output / self.T, dim=1)
        loss = (torch.sum(-softmax_pred_T * student_outout)) / (B * W * H)

        return loss