import torch
import torch.nn as nn
import torch.nn.init as init

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.d = 64
        self.l = 8
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
        # residual connection
        y = self.conv_output(y) + x
        return y
    
class PGDenoisingNetwork(nn.Module):
    def __init__(self):
        super(PGDenoisingNetwork, self).__init__()
        self.CNN = CNN()
        self.C0 = nn.Parameter(torch.zeros(1))
        self.Ck = nn.Parameter(torch.zeros(1))

        # initialize
        self.Ck.data.fill_(2.0)
        
    def forward(self, x):
        xk = x.clone()
        # Iterations
        for k in range(4):
            ak = self.C0 * torch.pow(self.Ck, -k)
            xk = (ak * x + self.CNN(xk)) / (ak + 1.0)
        return xk

if __name__ == '__main__':
    model = PGDenoisingNetwork()
    x = torch.randn(2, 3, 180, 180)
    print(model(x).size())
