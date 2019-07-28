import torch
import torch.nn as nn


class Conv5(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Conv5, self).__init__()
        self.conv_loc = nn.Conv2d(64,64, kernel_size=2, padding=2, dilation=2, stride=1)
        self.conv_gather = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1, stride=1)
        weight_map = torch.ones(3,3)
        weight_map[0,0] = weight_map[2,2] = weight_map[0,2] = weight_map[2,0] = 1/2
        self.conv_loc.weight = nn.Parameter(self.conv_gather.weight * weight_map)

    def forward(self, fea):
        fea = self.conv_loc(fea)
        fea = self.conv_gather(fea)
        return fea


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)