import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.utils as vutils
import tools
import numpy as np
import cv2

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
        
def complex_multiplication(t1, t2):
    real1, imag1 = t1.transpose(0,4)
    real2, imag2 = t2.transpose(0,4)
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = 0).transpose(0,4)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.d = 64
        self.l = 3
        self.conv = nn.ModuleList()
        self.batchnorm = nn.ModuleList()

        # convolution
        self.conv_input = nn.Conv2d(3, self.d, 3, 1, 1, bias=False)
        for i in range(self.l):
            self.conv.append(nn.Conv2d(self.d, self.d, 3, 1, 1, bias=False))
        self.conv_output = nn.Conv2d(self.d, 3, 3, 1, 1, bias=False)
                             
        # activation function
        self.activation = nn.ReLU(True)

        # batchnorm
        for i in range(self.l+1):
            self.batchnorm.append(nn.BatchNorm2d(self.d))

        self.apply(weights_init)

    def forward(self, x):
        y = self.activation(self.batchnorm[0](self.conv_input(x)))
        for i in range(self.l):
            y = self.activation(self.batchnorm[i+1](self.conv[i](y)))
        y = self.conv_output(y) + x
        return y
    
class PGDeblurringNetwork(nn.Module):
    def __init__(self):
        super(PGDeblurringNetwork, self).__init__()
        self.CNN = CNN()
        self.C0 = nn.Parameter(torch.zeros(1))
        self.Ck = nn.Parameter(torch.zeros(1))
        
        k = tools.fspecial('disk', 7)
        kpad = np.zeros((256, 256))
        kpad[121:136, 121:136] = k
        kshift = np.fft.fftshift(kpad)
        kshift = torch.from_numpy(kshift).float()
        self.K = torch.rfft(kshift.unsqueeze(0).unsqueeze(0), 2, onesided=False)
        self.K2 = torch.pow(torch.norm(self.K, dim=4).unsqueeze(4), 2)
        self.one = torch.ones(self.K2.shape)
        if torch.cuda.is_available():
            self.K2 = self.K2.cuda()
            self.K = self.K.cuda()
            self.one = self.one.cuda()

        # initialize
        self.Ck.data.fill_(2.0)
        self.C0.data.fill_(1000.0)
        
    def forward(self, x):
        kTyfft = tools.complex_multiplication(torch.rfft(x, 2, onesided=False), self.K)
        if torch.cuda.is_available():
            kTyfft = kTyfft.cuda()
        xk = x.clone()
        for k in range(8):
            ak = self.C0 * torch.pow(self.Ck, -k)
            xk = ak * kTyfft + torch.rfft(self.CNN(xk), 2, onesided=False)
            xk = xk / (ak * self.K2 + self.one)
            xk = torch.irfft(xk, 2, onesided=False)
        return xk

if __name__ == '__main__':
    model = PGDeblurringNetwork()
    x = torch.randn(2, 3, 180, 180)
    print(model(x).size())
