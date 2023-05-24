from torch import nn
import torchvision
import torch
import torch.nn.functional as F
from nets.utils import TrippleConv, multi_scale_aspp, MHSA
from torchvision.models import resnet, densenet201
from nets.deformable_conv import DeformConv2d

import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
from torchvision.models import resnet
import torch
from torchvision import models
from torch import nn

from functools import partial


nonlinearity = partial(F.relu, inplace=True)


def softplus_feature_map(x):
    return torch.nn.functional.softplus(x)

#------------------------------------------------------------------------------

class backbone(nn.Module):
    def __init__(self):
        super().__init__()

        baselayers = torchvision.models.densenet201(pretrained=True, progress=True)
        self.custom_model = nn.Sequential(*list(baselayers.features.children())[:-4])
        #print(self.custom_model)

    def forward(self, input):
        output = self.custom_model(input)
        return output
#------------------------------------------------
class dilated_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.convd1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=1, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.convd2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), stride=1, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

    def forward(self, inputs):
        x = self.convd1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.convd2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        return x
#---------------------------------------------------
#------------------------------------------------
class deform_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.deform1 = DeformConv2d(in_c, out_c, 3, padding=1, modulation=True)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)


    def forward(self, inputs):
        x = self.deform1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x
#---------------------------------------------------
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, dilation):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), stride=2, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3), stride=1, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
#---------------------------------------------
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
       # self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=True)
       # self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(in_c, out_c)

    def forward(self, inputs):
        up = self.up(inputs)
        x =self.conv(up)


        #x = torch.cat([x, skip], axis=1)
        #x = self.conv(x)
        return x
#------------------------------------------
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()
       # self.upsampling2 = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
###########################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
#--------------------------------------------------
class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
#------------------------------------------------------

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
#------------------------------------------------------------------------

class PAM_Module(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(PAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.softplus_feature = softplus_feature_map
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.softplus_feature(Q).permute(-3, -1, -2)
        K = self.softplus_feature(K)

        KV = torch.einsum("bmn, bcn->bmc", K, V)

        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)

        # weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class CAM_Module(nn.Module):
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)

        out = self.gamma * out + x
        return out

#----------------------------
class PAM_CAM_Layer(nn.Module):
    def __init__(self, in_ch):
        super(PAM_CAM_Layer, self).__init__()
        self.PAM = PAM_Module(in_ch)
        self.CAM = CAM_Module()

    def forward(self, x):
        return self.PAM(x) + self.CAM(x)


class DMCSNet(nn.Module):
    def __init__(self, init_weights=False,r=16):
        super(DMCSNet, self).__init__()

        # Encoder
        self.block1 = backbone() # torch.Size([1, 256, 64, 64])

        # Stream 1
        self.e1 = encoder_block(3,64,2) #
        self.e2 = encoder_block(64,128,2) #
        self.e3 = encoder_block(128, 256,2)  #
        self.e4 = encoder_block(256, 512,2)

        # Stream 2
        self.e5 = encoder_block(3, 64, 4)  #
        self.e6 = encoder_block(64, 128, 4)  #
        self.e7 = encoder_block(128, 256, 4)  #
        self.e8 = encoder_block(256, 512, 4)

        # self.e9 = encoder_block(3, 32, 1)  #
        # self.e10 = encoder_block(32, 64, 1)  #
        # self.e11 = encoder_block(64, 128, 1)  #
        # self.e12 = encoder_block(128, 256, 1)

        self.sa = SpatialAttention()
        self.se = SE_Block(1280, r)


        # # Stream 3
        # self.e9 = encoder_block(3, 32, 6)  #
        # self.e10 = encoder_block(32, 64, 6)  #
        # self.e11 = encoder_block(64, 128, 6)  #
        # self.e12 = encoder_block(128, 256, 6)


        # # # Stream 2 Dilated convoltion
        # self.e4 = dilated_block(256, 256)
        # self.e5 = dilated_block(256, 512)    # torch.Size([1, 1024, 16, 16])
        # self.e6 = dilated_block(512, 1024)
        #
      #   self.ca = ChannelAttention(256)
      # #  self.multihead = MHSA(768, width=32, height=32, heads=4)
      #  # self.sa = SpatialAttention()
      #
      #   self.conv_1by1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)

        # # Stream 3 Dilated covolution
        # # self.e6 = dilated_block(256, 256)
        # # self.e7 = dilated_block(256, 512)
        # #
        # # Attention mechanisms
        #
        # # # Decoder
        self.d1 = decoder_block(1280,512)  # 2048, 16, 16
        self.d2 = decoder_block(512, 256)  # 256, 32, 32
        self.d3 = decoder_block(256, 128)  # 128, 64, 64
        self.d4 = decoder_block(128, 64)   # 64, 256, 256
       # self.d5 = decoder_block(64, 32)  # 64, 256, 256
        # # #
        # # #
        # # # # Classification
        self.outc = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.last_activation = nn.Sigmoid()
        # if init_weights:
        #     print('initialize weights...')
        #     self.apply(self._initialize_weights)


    def forward(self, input):
            # Encoder
        s1 = self.block1(input)  # torch.Size([1, 256, 16, 16])


        # stream 1
        e1 = self.e1(input)
        e2 = self.e2(e1) # torch.Size([1, 512, 16, 16])
        e3 = self.e3(e2)
        e4 = self.e4(e3)
      #  e4 = self.sa(e4) * e4

     #
     #    # stream 2
        e5 = self.e5(input)
        e6 = self.e6(e5)  # torch.Size([1, 512, 16, 16])
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        # stream 3
        # e9 = self.e9(input)
        # e10 = self.e10(e9)  # torch.Size([1, 512, 16, 16])
        # e11 = self.e11(e10)
        # e12 = self.e12(e11)
     #
        encoder_out_1 = torch.cat((e4, e8), 1)
       # encoder_out_1 = self.sa(encoder_out_1) * encoder_out_1


        encoder_out_2 = torch.cat((encoder_out_1, s1), 1)

     #
     #
       # encoder_cag_res = torch.cat((encoder_out_2, s1), 1)
        out_cag = self.se(encoder_out_2)

        d1 = self.d1(out_cag)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        if self.last_activation is not None:
            output = self.last_activation(self.outc(d4))

        return output
    #---------------------------------------------------------
    # def init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform(m.weight)
    #         m.bias.data.fill_(0.01)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

input = torch.randn((1,3,256,256))
model = DMCSNet()
num_trainable_params = count_parameters(model)
print("Number of trainable parameters: {}".format(num_trainable_params))
# # # # # Initialize the weights
# # # # #model.apply(weights_init)
