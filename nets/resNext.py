import torch
import torch.nn as nn

class ResNextBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, gChannel):
        super(ResNextBlock, self).__init__()
        self.resNext = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=gChannel, kernel_size=(1, 1), stride=1, padding=0),
            nn.BatchNorm2d(gChannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=gChannel, out_channels=gChannel, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(gChannel),
            nn.ReLU(inplace=True),
        )
        self.conv1x1 = nn.Conv2d(in_channels=gChannel * 4, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=0)
        self.conv1x1Input = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=0)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.conv1x1Input(x)
        out1 = self.resNext(x)
        out2 = self.resNext(x)
        out3 = self.resNext(x)
        out4 = self.resNext(x)
        result = torch.cat([out1, out2, out3, out4], dim=1)
        result = self.conv1x1(result)
        result += residual
        return result



