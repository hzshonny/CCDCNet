# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from util.util import vgg_preprocess
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
from timm.models.layers import DropPath

import re

from models.networks.base_network import BaseNetwork
import numpy as np
'''363'''
class FF_Module(nn.Module):
    def __init__(self, channel):
        super(FF_Module, self).__init__()
        self.conv = nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(channel//2, channel // 2, kernel_size=3, padding=1, bias=False)
        self.actvn = nn.LeakyReLU(0.2, False)
        self.inorm = nn.InstanceNorm2d(channel // 2)
    def forward(self, x1,x2):
        x=torch.concat([x1,x2],dim=1)
        x=self.actvn(self.inorm(self.conv(x)))
        x=self.inorm(self.conv1(x))
        return x

'''395'''
# class FF_Module(nn.Module):
#     def __init__(self, channel):
#         super(FF_Module, self).__init__()
#         self.conv = nn.Conv2d(channel, channel // 2, kernel_size=3, padding=1, bias=False)
#         # self.conv1 = nn.Conv2d(channel//2, channel // 2, kernel_size=3, padding=1, bias=False)
#         self.actvn = nn.LeakyReLU(0.2, False)
#         self.inorm = nn.InstanceNorm2d(channel // 2)
#     def forward(self, x1,x2):
#         x=torch.concat([x1,x2],dim=1)
#         x=self.actvn(self.inorm(self.conv(x)))
#         # x=self.conv1(x)
#         return x

class AdainBlock(nn.Module):
    def __init__(self, dim, dimin, use_bias=False):
        super(AdainBlock, self).__init__()
        self.epsilon = 1e-8
        self.norm = AdaIN()
        self.styleModulator = nn.Linear(dimin, 2*dim)
        self.dim = dim
        with torch.no_grad():
            self.styleModulator.weight *= 0.25
            self.styleModulator.bias.data.fill_(0)

    def forward(self, x, y):
        # Adapt style
        batchSize, nChannel, width, height = x.size()
        styleY = self.styleModulator(y) # torch.Size([4, 64])
        # print(y.shape)
        y_var = styleY[:, :self.dim].view(batchSize, self.dim, 1, 1)
        y_mean = styleY[:, self.dim:].view(batchSize, self.dim, 1, 1)
        out = self.norm(x, y_mean, y_var)
        return out

class AdaIN(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y_mean, y_var):
        # x: N x C x W x H
        size = x.size()
        assert (len(size) == 4)
        b, c = x.shape[:2]
        feat_x = x.view(b, c, -1)
        x_mean = feat_x.mean(dim=2).view(b, c, 1, 1)
        varx = torch.clamp((feat_x * feat_x).mean(dim=2).view(b, c, 1, 1) - x_mean * x_mean, min=0)
        varx = torch.rsqrt(varx + self.epsilon)
        x = (x - x_mean) * varx
        # print(x.shape)
        # print(y_var.shape)
        return x * y_var + y_mean

# try:
#     import apex
#     from apex import amp
# except:
#     print('apex not found')
#     pass

class EqualLR:
    def __init__(self, name):
        self.name = name
    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * np.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        return fn
    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module

class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc, PONO=False, use_apex=False):
        super().__init__()
        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        self.pad_type = 'nozero'

        if PONO:
            self.param_free_norm = PositionalNorm2d
        elif param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'syncbatch':
            
            # if use_apex:
            #     self.param_free_norm = apex.parallel.SyncBatchNorm(norm_nc, affine=False)
            # else:
            #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
            self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)

        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        if self.pad_type != 'zero':
            self.mlp_shared = nn.Sequential(
                nn.ReflectionPad2d(pw),
                nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=0),
                nn.ReLU()
            )
            self.pad = nn.ReflectionPad2d(pw)
            self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)
            self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=0)
        else:
            self.mlp_shared = nn.Sequential(
                    nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
                    nn.ReLU()
                )
            self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap, similarity_map=None):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        if self.pad_type != 'zero':
            gamma = self.mlp_gamma(self.pad(actv))
            beta = self.mlp_beta(self.pad(actv))
        else:
            gamma = self.mlp_gamma(actv)
            beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta

        return out


class AdaptiveFeatureGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        # TODO: kernel=4, concat noise, or change architecture to vgg feature pyramid
        super().__init__()
        self.opt = opt
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(opt.spade_ic, ndf, kw, stride=1, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

        nf = opt.ngf

        self.head_0 = SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)
        # if opt.adaptor_nonlocal:
        #     self.attn = Attention(8 * nf, False)
        self.G_middle_0 = SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)
        self.G_middle_1 = SPADEResnetBlock(8 * nf,8 * nf, opt, use_se=opt.adaptor_se)


    def forward(self, input, seg):
        x = self.layer1(input)
        x1=x
        '''---添加coordAttention---'''
        # x = self.catten(x)
        x = self.layer2(self.actvn(x))
        x2=x
        x = self.layer3(self.actvn(x))
        x3=x
        x = self.layer4(self.actvn(x))
        '''---添加coordAttention---'''
        # x = self.catten1(x)

        x = self.head_0(x, seg)
        # x = self.catten1(x)
        # print("7", x.shape) # 7 torch.Size([2, 512, 64, 64])
        '''---没用'''
        # if self.opt.adaptor_nonlocal:
        #     x = self.attn(x)
        # print("8", x.shape) # 8 torch.Size([2, 512, 64, 64])
        x = self.G_middle_0(x, seg)
        # x = self.catten1(x)
        # print("9", x.shape) # torch.Size([2, 512, 64, 64])
        x = self.G_middle_1(x, seg)
        # print("10", x.shape) # 10 torch.Size([2, 256, 64, 64])
        '''---添加coordAttention---'''
        # x = self.catten2(x)


        # print("11", x.shape) # 11 torch.Size([2, 256, 64, 64])
        return [x,x3,x2,x1]

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.opt = opt
        self.pad_type = 'nozero'
        self.use_se = use_se

        # create conv layers
        if self.pad_type != 'zero':
            self.pad = nn.ReflectionPad2d(dilation)
            self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0, dilation=dilation)
            self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0, dilation=dilation)
        else:
            self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
            self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            if opt.eqlr_sn:
                self.conv_0 = equal_lr(self.conv_0)
                self.conv_1 = equal_lr(self.conv_1)
                if self.learned_shortcut:
                    self.conv_s = equal_lr(self.conv_s)
            else:
                self.conv_0 = spectral_norm(self.conv_0)
                self.conv_1 = spectral_norm(self.conv_1)
                if self.learned_shortcut:
                    self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        if 'spade_ic' in opt:
            ic = opt.spade_ic
        else:
            # ic = 0 + (3 if 'warp' in opt.CBN_intype else 0) + (opt.semantic_nc if 'mask' in opt.CBN_intype else 0)
            ic=opt.semantic_nc

        self.norm_0 = SPADE(spade_config_str, fin, ic, PONO=opt.PONO, use_apex=opt.apex)
        self.norm_1 = SPADE(spade_config_str, fmiddle, ic, PONO=opt.PONO, use_apex=opt.apex)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, ic, PONO=opt.PONO, use_apex=opt.apex)

        # if use_se:
        #     self.se_layar = SELayer(fout)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg1):
        # print("x",x.shape) # torch.Size([2, 512, 64, 64])
        # print("seg1",seg1.shape) # seg1 torch.Size([2, 2, 256, 256])
        x_s = self.shortcut(x, seg1)  # x_ torch.Size([2, 512, 64, 64])
        # print("x_",x_s.shape)
        if self.pad_type != 'zero':
            dx = self.conv_0(self.pad(self.actvn(self.norm_0(x, seg1))))
            dx = self.conv_1(self.pad(self.actvn(self.norm_1(dx, seg1))))
            if self.use_se:
                dx = self.se_layar(dx)
        else:
            dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
            dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
            if self.use_se:
                dx = self.se_layar(dx)
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)



def get_nonspade_norm_layer(opt, norm_type='instance'):
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
        else:
            subnorm_type =norm_type
        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)
        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)
        return nn.Sequential(layer, norm_layer)
    return add_norm_layer

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super(MappingNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(self.kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, features):

        z = self.avgpool(features).view(features.shape[0], -1)
        mapping_codes = self.network(z)
        return mapping_codes

    def kaiming_leaky_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

def PositionalNorm2d(x, epsilon=1e-8):
    # x: B*C*W*H normalize in C dim
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ResidualBlock(nn.Module):
    def __init__(self, dim, ks=3):
        super(ResidualBlock, self).__init__()
        self.relu = nn.PReLU()
        self.model = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, stride=(1, 1), padding_mode='reflect'),
            nn.InstanceNorm2d(dim),
            self.relu,
            nn.Conv2d(dim, dim, kernel_size=(ks, ks), padding=ks // 2, stride=(1, 1), padding_mode='reflect'),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        out = self.relu(x + self.model(x))
        return out



class VGG19_feature_color_torchversion(nn.Module):
    """
    NOTE: there is no need to pre-process the input 
    input tensor should range in [0,1]
    """
    def __init__(self, pool='max', vgg_normal_correct=False, ic=3):
        super(VGG19_feature_color_torchversion, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct

        self.conv1_1 = nn.Conv2d(ic, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys, preprocess=True):
        ''' 
        NOTE: input tensor should range in [0,1]
        '''
        out = {}
        if preprocess:
            x = vgg_preprocess(x, vgg_normal_correct=self.vgg_normal_correct)
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]
