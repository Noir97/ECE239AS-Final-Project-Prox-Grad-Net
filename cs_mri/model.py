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
        self.l = 5
        self.conv = nn.ModuleList()
        self.batchnorm = nn.ModuleList()

        # convolution
        self.conv_input = nn.Conv2d(1, self.d, 3, 1, 1, bias=False)
        for i in range(self.l):
            self.conv.append(nn.Conv2d(self.d, self.d, 3, 1, 1, bias=False))
        self.conv_output = nn.Conv2d(self.d, 1, 3, 1, 1, bias=False)
                             
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
        y = self.conv_output(y)
        return y
    
class CS_MRI_Network(nn.Module):
    def __init__(self):
        super(CS_MRI_Network, self).__init__()
        self.CNN = CNN()
        
    def forward(self, x):
        xk = torch.ifft(x, 2)
        for k in range(8):
            xkk = self.CNN(xk[:,:,:,:,0]).view(x.shape[0],1,256,256,1)
            zk = torch.fft(xk+torch.cat((xkk, torch.zeros(xkk.shape).cuda()), 4), 2)
            xk = torch.ifft(x + (x==0).float()*zk, 2)
        return xk

if __name__ == '__main__':
    model = CS_MRI_Network()
    x = torch.randn(2, 3, 180, 180)
    print(model(x).size())