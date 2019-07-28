import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from modules.convs import InvertedResidual
from models.fastscnn import PSPModule

class MyModel(nn.Module):
    def __init__(self, num_classes=19):
        super(MyModel, self).__init__()
        self.head = RtHead_LargeKernel()
        self.context = RtContext()
        self.classifier = RtClassifer(num_classes=num_classes)

    def forward(self, fea):
        # h, w = fea.size()[-2:]
        low = self.head(fea)
        fea = self.context(low)
        fea = self.classifier(fea, low)
        # fea = interpolate(fea, size=(h,w), mode='bilinear', align_corners=True)
        return fea


class RtHead_LargeKernel(nn.Module):
    def __init__(self):
        super(RtHead_LargeKernel, self).__init__()
        self.conv1 = self._make_conv_block(dim_in=3, dim_out=16, kernel_size=7, pad=3, stride=2)
        self.conv2 = self._make_conv_block(dim_in=16, dim_out=32, kernel_size=5, pad=2, stride=2)
        self.conv3 = self._make_conv_block(dim_in=32, dim_out=64, kernel_size=3, pad=1, stride=2)

    @staticmethod
    def _make_conv_block(dim_in, dim_out, kernel_size, pad, stride):
        return nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=pad, stride=stride),
            nn.BatchNorm2d(dim_out),
            nn.ReLU()
        )

    def forward(self, fea):
        fea = self.conv1(fea)
        fea = self.conv2(fea)
        fea = self.conv3(fea)
        return fea


class RtContext(nn.Module):
    def __init__(self):
        super(RtContext, self).__init__()
        self.convs = nn.Sequential(
            InvertedResidual(inp=64, oup=96, stride=2, expand_ratio=6),
            # InvertedResidual(inp=64, oup=96, stride=1, expand_ratio=6),
            InvertedResidual(inp=96, oup=128, stride=2, expand_ratio=6),
            # InvertedResidual(inp=96, oup=128, stride=1, expand_ratio=6)
        )

        # self.block1 = InvertedResidual(inp=64, oup=64, stride=1, expand_ratio=6)
        # self.block2 = InvertedResidual(inp=64, oup=96, stride=2, expand_ratio=6)
        # self.block3 = InvertedResidual(inp=96, oup=96, stride=1, expand_ratio=6)
        # self.block4 = InvertedResidual(inp=96, oup=128, stride=2, expand_ratio=6)
        # self.ppm = PSPModule(128, 128)
        # self.block5 = InvertedResidual(inp=128, oup=128, stride=1, expand_ratio=6)

    def forward(self, fea):
        # fea = self.block1(fea)
        # fea = self.block2(fea)
        # fea = self.block3(fea)
        # fea = self.block4(fea)
        fea = self.convs(fea)
        # fea = self.ppm(fea)
        # fea = self.block5(fea)
        # fea = self.block6(fea)
        # fea = self.block7(fea)
        return fea


class RtClassifer(nn.Module):
    def __init__(self, num_classes):
        super(RtClassifer, self).__init__()
        self.low_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.high_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=3, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )

    def forward(self, fea, low_level_fea):
        low = self.low_conv(low_level_fea)
        high = interpolate(self.high_conv(fea), low.size()[-2:], mode='bilinear', align_corners=True)
        fea = self.classifier(low + high)
        return fea


if __name__ == '__main__':

    # m = RtHead_LargeKernel().cuda()
    # m = RtContext().cuda(1)
    # a = torch.randn(1, 64, 128, 256).cuda(1)

    m = MyModel().cuda(1)
    print(m)
    # m = RtHead_LargeKernel().cuda(1)
    # m = RtContext().cuda(1)
    # m = RtClassifer(num_classes=19).cuda(1)
    a = torch.randn(1, 3, 1024, 2048).cuda(1)
    # b = torch.randn([1, 64, 128, 256]).cuda(1)
    # c = torch.randn(1, 128, 32, 64).cuda(1)
    import time
    st = time.time()

    for i in range(100):
        # y = m(c, b)
        y = m(a)

    delta = time.time() - st
    print('Using %f' % delta)

    print(' %f itr/s' % (100 / delta))
    print(y.size())


