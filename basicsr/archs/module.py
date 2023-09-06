import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
        
class OrthorTransform(nn.Module):
    def __init__(self, c_dim, feat_hw, groups): #feat_hw: width or height (let width == heigt)
        super(OrthorTransform, self).__init__()

        self.groups = groups
        self.c_dim = c_dim
        self.feat_hw = feat_hw
        self.weight = nn.Parameter(torch.randn(1, feat_hw, c_dim))
        self.opt_orth = optim.Adam(self.parameters(), lr=1e-3, betas=(0.5, 0.99))    
    def forward(self, feat):
        pred = feat * self.weight.expand_as(feat)
        return pred, self.weight.view(self.groups, -1)

class CodeReduction(nn.Module):
    def __init__(self, c_dim, feat_hw, blocks = 4, prob=False):
        super(CodeReduction, self).__init__()
        self.body = nn.Sequential(
            nn.Linear(c_dim, c_dim*blocks),
            nn.LeakyReLU(0.2, True)
        )
        self.trans = OrthorTransform(c_dim=c_dim*blocks, feat_hw=feat_hw, groups = blocks)
        self.leakyrelu = nn.LeakyReLU(0.2, True)
    def forward(self, feat):
        feat = self.body(feat)
        feat, weight = self.trans(feat)
        feat = self.leakyrelu(feat)
        return feat, weight