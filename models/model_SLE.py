import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet34_Weights

class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1

class Network_SLE(nn.Module):
    def __init__(self, input_channels=9):
        super(Network_SLE, self).__init__()
        self.base_model = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.base_layers = list(self.base_model.children())
        self.encode1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2],
        )                                                     # [b, 64, h/2, w/2]
        self.encode2 = nn.Sequential(*self.base_layers[3:5])  # [b, 64, h/4, w/4]
        self.encode3 = self.base_layers[5]                    # [b, 128, h/8, w/8]
        self.encode4 = self.base_layers[6]                    # [b, 256, h/16, w/16]
        self.encode5 = self.base_layers[7]                    # [b, 512, h/32, w/32]

        self.decode5 = Decoder(in_channels=512, middle_channels=256+256, out_channels=256)
        self.decode4 = Decoder(in_channels=256, middle_channels=128+128, out_channels=128)
        self.decode3 = Decoder(in_channels=128, middle_channels=64+64, out_channels=64)
        self.decode2 = Decoder(in_channels=64, middle_channels=64+64, out_channels=64)
        self.decode1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(in_channels=64, out_channels=7, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skeleton, densepose, parse_swap):
        input = torch.cat((skeleton, densepose, parse_swap), axis=1)    # [b, 1+1+7, h, w]
        e1 = self.encode1(input)            # [b,64,h/2,w/2]
        e2 = self.encode2(e1)               # [b,64,h/4,w/4]
        e3 = self.encode3(e2)               # [b,128,h/8,w/8]
        e4 = self.encode4(e3)               # [b,256,h/16,w/16]
        f = self.encode5(e4)                # [b,512,h/32,w/32]

        d4 = self.decode5(f, e4)            # [b,256,h/16,w/16] 
        d3 = self.decode4(d4, e3)           # [b,128,h/8,w/8] 
        d2 = self.decode3(d3, e2)           # [b,64,h/4,w/4]  
        d1 = self.decode2(d2, e1)           # [b,64,h/2,w/2]
        d0 = self.decode1(d1)       
        parse7_t = self.conv_last(d0)       # [b,7,h,w]
        parse7_t = self.sigmoid(parse7_t)   # [b,7,h,w]
        return parse7_t

if __name__ == '__main__':
    skeleton = torch.randn(4, 1, 512, 384).cuda()
    densepose = torch.randn(4, 1, 512, 384).cuda()
    parse_swap = torch.randn(4, 7, 512, 384).cuda()
    
    model = Network_SLE().cuda()
    
    parse = model(skeleton, densepose, parse_swap)
    print("parse:", parse.shape)

    