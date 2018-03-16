import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DoubleConvBN(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConvBN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class FirstConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstConv, self).__init__()
        self.conv = DoubleConvBN(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x


class ContractingPathConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContractingPathConv, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConvBN(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class ExpansivePathConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExpansivePathConv, self).__init__()
        self.upscale = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, output_padding=0)
        self.conv = DoubleConvBN(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upscale(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = FirstConv(n_channels, 64)
        self.conv1 = ContractingPathConv(64, 128)
        self.conv2 = ContractingPathConv(128, 256)
        self.conv3 = ContractingPathConv(256, 512)
        self.conv4 = ContractingPathConv(512, 1024)
        self.up1 = ExpansivePathConv(1024, 512)
        self.up2 = ExpansivePathConv(512, 256)
        self.up3 = ExpansivePathConv(256, 128)
        self.up4 = ExpansivePathConv(128, 64)
        self.outc = FinalConv(64, n_classes)

    def forward(self, x):
        image_size = x.size()[2:]
        x1 = self.inc(x)
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = F.upsample(x, image_size, mode='bilinear')
        return x


if __name__ == '__main__':
    net = UNet(3, 11)
    w = 480
    h = 480
    a = Variable(torch.FloatTensor(1, 3, w, h))
    b = net(a)
    print(a.size(), b.size())
