import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, (3, 3), padding=1)

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = x + output
        #        print('ResBlock ouput:{}'.format(output.dtype))
        return output


class UpSampling(torch.autograd.Function):

    def forward(self, x):
        n, ch, h, w = x.shape
        output = Variable(torch.zeros(n, int(ch / 4), h * 2, w * 2))
        output=output.to(device=x.device,dtype=x.dtype)
        for r in range(h):
            for c in range(w):
                output[:, :, r * 2:r * 2 + 2, c * 2:c * 2 + 2] = x[:, :, r, c].reshape(-1, int(ch / 4), 2, 2)
        return output

    def backward(self, grad_out):
        n, _, h, w = grad_out.shape
        grad_in = Variable(torch.zeros(n, 64, int(h / 2), int(w / 2)))
        grad_in=grad_in.to(device=grad_out.device,dtype=grad_out.dtype)
        for r in range(int(h / 2)):
            for c in range(int(w / 2)):
                grad_in[:, :, r, c] = grad_out[:, :, r * 2:r * 2 + 2, c * 2:c * 2 + 2].reshape(-1, 64)

        return grad_in


class UpSamplingBlock(nn.Module):
    def __init__(self):
        super(UpSamplingBlock, self).__init__()
        self.conv1 = nn.Conv2d(16, 64, (3, 3), padding=1)
        self.relu = nn.ReLU()
        self.upsampling = UpSampling()

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)
        output = self.upsampling(output)
        #        print('Upsampling Block output:{}'.format(output.dtype))
        return output


class CNN(nn.Module):
    def __init__(self, input_channel=1):
        super(CNN, self).__init__()
        self.path_d8 = nn.Sequential(
            nn.Conv2d(input_channel, 16, (3, 3), 8, 1),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            UpSamplingBlock(),
            UpSamplingBlock(),
            UpSamplingBlock(),
            nn.Conv2d(16, 16, (3, 3), padding=1)
        )
        self.path_d4 = nn.Sequential(
            nn.Conv2d(input_channel, 16, (3, 3), 4, 1),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            UpSamplingBlock(),
            UpSamplingBlock(),
            nn.Conv2d(16, 16, (3, 3), padding=1)
        )
        self.path_d2 = nn.Sequential(
            nn.Conv2d(input_channel, 16, (3, 3), 2, 1),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            UpSamplingBlock(),
            nn.Conv2d(16, 16, (3, 3), padding=1)
        )
        self.path_d1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, (3, 3), 1, 1),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            nn.Conv2d(16, 16, (3, 3), padding=1)
        )
        self.tail = nn.Conv2d(64, 1, (3, 3), padding=1)

    def forward(self, x):
        output_d8 = self.path_d8(x)
        output_d4 = self.path_d4(x)
        output_d2 = self.path_d2(x)
        output_d1 = self.path_d1(x)

        output = torch.cat((output_d1, output_d2, output_d4, output_d8), 1)

        output = self.tail(output)
        #        print('CNN output:{}'.format(output.dtype))
        return output


if __name__ == '__main__':
    x = torch.randn(1, 1, 64, 64)
    model = CNN()
    model(x)

