import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision.models import ResNet18_Weights

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim):
        super(MLP, self).__init__()

        self.l1 = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(True)
        )
        self.l3 = nn.Sequential(
            nn.Linear(dim, output_dim),
            nn.ReLU(True)
        )

    def forward(self, style):
        s1 = self.l1(style)
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        return s3

class StyleFusionBlock(nn.Module):
    def __init__(self):
        super(StyleFusionBlock, self).__init__()

        self.fusion_block1 = nn.Sequential(
            nn.Conv2d(512, 512, 5, 1, 2),
            AdaptiveInstanceNorm2d(512),
            nn.ReLU(True)
        )

        self.fusion_block2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            AdaptiveInstanceNorm2d(512),
            nn.ReLU(True)
        )

    def forward(self, feature):
        x = feature
        p1 = self.fusion_block1(feature)
        p2 = self.fusion_block2(p1)
        return p2 + x

class StyleFusions(nn.Module):
    def __init__(self):
        super(StyleFusions, self).__init__()

        self.style_fusion_block1 = StyleFusionBlock()
        self.style_fusion_block2 = StyleFusionBlock()
        self.style_fusion_block3 = StyleFusionBlock()
        self.style_fusion_block4 = StyleFusionBlock()
        self.style_fusion_block5 = StyleFusionBlock()

    def forward(self, feature):
        f1 = self.style_fusion_block1(feature)
        f2 = self.style_fusion_block2(f1)
        f3 = self.style_fusion_block3(f2)
        f4 = self.style_fusion_block4(f3)
        f5 = self.style_fusion_block5(f4)
        return f5

class StyleEncoder(nn.Module):
    def __init__(self, in_dim=3): 
        super(StyleEncoder, self).__init__()

        self.base_model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_layers = list(self.base_model.children())
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2],
        )
        self.encode2 = nn.Sequential(*self.base_layers[3:5])
        self.encode3 = self.base_layers[5]
        self.encode4 = self.base_layers[6]
        self.encode5 = self.base_layers[7]

    def forward(self, cloth):
        p1 = self.encode1(cloth)        # [b, 64,  h/2, w/2]
        p2 = self.encode2(p1)           # [b, 256, h/4, w/4]
        p3 = self.encode3(p2)           # [b, 256, h/8, w/8]
        p4 = self.encode4(p3)           # [b, 256, h/16, w/16]
        p5 = self.encode5(p4)           # [b, 256, h/32, w/32]
        style = p5.view(p5.size(0), -1) # [b, 256 * (h/32) * (w/32)]
        return style

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class Decoder(nn.Module):
    def __init__(self, n_upsample, dim, output_dim, activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class PoseEncoder(nn.Module):
    def __init__(self, in_dim=1):
        super(PoseEncoder, self).__init__()
        self.base_model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_layers = list(self.base_model.children())
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2],
        )
        self.encode2 = nn.Sequential(*self.base_layers[3:5])
        self.encode3 = self.base_layers[5]
        self.encode4 = self.base_layers[6]
        self.encode5 = self.base_layers[7]

    def forward(self, pose):
        p1 = self.encode1(pose)  # [b, 64, h/2, w/2]
        p2 = self.encode2(p1)           
        p3 = self.encode3(p2)           
        p4 = self.encode4(p3)           
        p5 = self.encode5(p4)    # [b, 512, h/32, w/32]
        return p5

class FeatureEncoder(nn.Module):
    def __init__(self, input_channels=6):
        super(FeatureEncoder, self).__init__()
        self.base_model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.base_layers = list(self.base_model.children())
        self.encode1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2],
        )
        self.encode2 = nn.Sequential(*self.base_layers[3:5])
        self.encode3 = self.base_layers[5]                    
        self.encode4 = self.base_layers[6]                    
        self.encode5 = self.base_layers[7]       

    def forward(self, input):
        e1 = self.encode1(input)     # [b,64,h/2,w/2]
        e2 = self.encode2(e1)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)
        f = self.encode5(e4)         # [b,512,h/32,w/32]
        return f

class Network_SCW(nn.Module):
    def __init__(self): 
        super(Network_SCW, self).__init__()

        # encoder for person
        self.pose_encoder = PoseEncoder(in_dim=3)
        self.style_encode = StyleEncoder(in_dim=3)
        # encoder for clothing
        self.style_fusions = StyleFusions()
        self.mlp = MLP(input_dim=512*16*12, output_dim=self.get_num_adain_params(self.style_fusions), dim=512)
        # decode
        self.dec = Decoder(n_upsample=5, dim=512, output_dim=4)
        self.sigmoid = nn.Sigmoid()
        # appearance flow
        self.flow_net = Decoder(n_upsample=5, dim=1024, output_dim=2)
        # activation
        self.tanh = nn.Tanh()

        # ==============================
        hidden_size=512
        num_attention_heads=8
        self.encoder1 = FeatureEncoder(input_channels=3)
        self.encoder2 = FeatureEncoder(input_channels=3)
        self.encoder3 = FeatureEncoder(input_channels=3)
        self.linear = nn.Linear(512*16*12, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 512*16*12)
        # cross-attention
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_attention_heads)

    def forward(self, skeleton, densepose, parse_preserve, cloth, mloth, cloth_norm):
        # -------- encode human representation -------------
        pose_inputs = torch.cat((skeleton, densepose, parse_preserve), axis=1)
        pose_feature = self.pose_encoder(pose_inputs)   # [b, 512, h/32, w/32]
        # -------- embed clothing style -------------
        style = self.style_encode(cloth)
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.style_fusions)
        decode_res = self.style_fusions(pose_feature)   # [b, 512, h/32, w/32]
        # ------------ decode ----------------
        coarse_wloth = self.dec(decode_res)
        coarse_wloth = self.sigmoid(coarse_wloth)
        gen_cloth = coarse_wloth[:,0:3,:,:]
        gen_mloth = coarse_wloth[:,3,:,:].unsqueeze(1)
        # ------------ cross-attention ------------
        q = self.encoder1(gen_cloth)
        k = self.encoder2(cloth_norm)
        v = self.encoder3(cloth_norm)
        q = q.view(q.size(0), -1)
        k = k.view(k.size(0), -1)
        v = v.view(v.size(0), -1)
        q = self.linear(q)
        k = self.linear(k)
        v = self.linear(v)
        cross_attention_output, _ = self.cross_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
        # ------------- cross_f ------------------------
        cross_f = self.linear2(cross_attention_output.squeeze(0))
        cross_f = cross_f.view(cross_f.size(0), 512, 16, 12)                    # [b, 512, h/32, w/32]
        # ------------- flow ---------------
        offset = self.flow_net(torch.cat([decode_res, cross_f], axis=1))        # [b,2,h,w]
        offset = offset.permute(0,2,3,1)                                        # [b,h,w,2]

        offset = self.tanh(offset)
        gridY = torch.linspace(-1, 1, steps = 512).view(1, -1, 1, 1).expand(1, 512, 384, 1)
        gridX = torch.linspace(-1, 1, steps = 384).view(1, 1, -1, 1).expand(1, 512, 384, 1)
        grid = torch.cat((gridX, gridY), dim=3).type(offset.type())
        grid = torch.repeat_interleave(grid, repeats=offset.shape[0], dim=0)
        flow = torch.clamp(offset + grid, min=-1, max=1)
        flow_cloth = F.grid_sample(cloth, flow, mode='bilinear', padding_mode='border', align_corners=True)
        flow_mloth = F.grid_sample(mloth, flow, mode='bilinear', padding_mode='zeros', align_corners=True)   

        warp_cloth = flow_cloth * gen_mloth + (1-gen_mloth)

        gen_mloth[gen_mloth > 0.1] = 1
        gen_mloth[gen_mloth <= 0.1] = 0
        warp_mloth = flow_mloth * gen_mloth

        return warp_cloth, warp_mloth

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                # print("m.num_features:", m.num_features)
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]

                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

if __name__ == "__main__":
    skeleton = torch.from_numpy(np.zeros((2,1,512,384)).astype(np.float32)).cuda()
    densepose = torch.from_numpy(np.zeros((2,1,512,384)).astype(np.float32)).cuda()
    parse_preserve = torch.from_numpy(np.zeros((2,1,512,384)).astype(np.float32)).cuda()
    cloth = torch.from_numpy(np.zeros((2,3,512,384)).astype(np.float32)).cuda()
    mloth = torch.from_numpy(np.zeros((2,1,512,384)).astype(np.float32)).cuda()
    cloth_norm = torch.from_numpy(np.zeros((2,3,512,384)).astype(np.float32)).cuda()

    model = Network_SCW().cuda()
    flow_cloth, flow_mloth = model(skeleton, densepose, parse_preserve, cloth, mloth, cloth_norm)
 
    print("flow_cloth:", flow_cloth.shape)   
    print("flow_mloth:", flow_mloth.shape)   


