import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import util.util as util

from models.networks.base_network import BaseNetwork
from models.networks.architecture import (SPADEResnetBlock, MappingNetwork, ResidualBlock,get_nonspade_norm_layer, \
        AdainBlock, FF_Module)
from models.networks.nceloss import BidirectionalNCE1

from models.networks.ops import dequeue_data, queue_data
from models.networks import calc_contrastive_loss

from models.networks.architecture import AdaptiveFeatureGenerator
from models.networks.cc_attention import CrissCrossAttention1


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class PatchSampleF(nn.Module):
    def __init__(self):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)

    def forward(self, feat, num_patches=64, patch_ids=None):
        # b c h w --> b h w c --> b hw c
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
        if patch_ids is not None:
            patch_id = patch_ids
        else:
            patch_id = torch.randperm(feat_reshape.shape[1], device=feat[0].device)
            patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
        x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
        x_sample = self.l2norm(x_sample)
        # return_feats.append(x_sample)
        return x_sample, patch_id


class ResidualBlock1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock1, self).__init__()
        self.padding1 = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.prelu(out)
        return out


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')

        parser.add_argument('--max_multi', type=int, default=8)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        opt.spade_ic = opt.semantic_nc
        self.adaptive_model_seg = AdaptiveFeatureGenerator(opt)
        opt.spade_ic = 3
        self.adaptive_model_img = AdaptiveFeatureGenerator(opt)
        del opt.spade_ic
        # if opt.weight_domainC > 0 and (not opt.domain_rela):
        #     self.domain_classifier = DomainClassifier(opt)

        if 'down' not in opt:
            opt.down = 4
        if opt.warp_stride == 2:
            opt.down = 2
        assert (opt.down == 2) or (opt.down == 4)
        self.down = opt.down
        self.feature_channel = 64
        self.in_channels = self.feature_channel * 4
        self.inter_channels = 256


        self.nceloss = BidirectionalNCE1()
        self.patch_sample = PatchSampleF()


        '''绝对位置编码
        Learned Positional Embedding方法是最普遍的绝对位置编码方法，该方法直接对不同的位置随机初始化一个 postion embedding，
        加到 word embedding 上输入模型，作为参数进行训练。'''
        pos_embed = nn.Parameter(torch.randn(
            1, self.feature_channel * 8 , opt.crop_size // 8, opt.crop_size // 8))
        self.register_parameter('pos_embed', pos_embed)

        self.layer = nn.Sequential(
            ResidualBlock1(self.feature_channel * 8, self.feature_channel * 8,kernel_size=3, padding=1, stride=1),
            ResidualBlock1(self.feature_channel * 8, self.feature_channel * 8,kernel_size=3, padding=1, stride=1),
            ResidualBlock1(self.feature_channel * 8, self.feature_channel * 8, kernel_size=3, padding=1, stride=1),
        )

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer5 = nn.Sequential(
            norm_layer(nn.Conv2d(opt.ngf * 4 * 2, opt.ngf * 4 * 2, kernel_size=3, stride=2, padding=1)),
            ResidualBlock(opt.ngf * 4 * 2),
        )

        self.recurrence=2
        self.cc_attn1 = CrissCrossAttention1(self.feature_channel * 8)


        nf = opt.ngf


        self.up_0 = SPADEResnetBlock(8 * nf,8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8* nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)
        self.up_4 = SPADEResnetBlock(1 * nf, 1 * nf, opt)

        self.conv0 = nn.Sequential(
        norm_layer(nn.Conv2d(nf * 16, nf * 8, kernel_size=3, padding=1, bias=False)),
        nn.LeakyReLU(0.2, False))
        self.conv1 = nn.Sequential(
        norm_layer(nn.Conv2d(nf * 16, nf * 8, kernel_size=3, padding=1, bias=False)),
        nn.LeakyReLU(0.2, False))
        self.conv2 = nn.Sequential(
        norm_layer(nn.Conv2d(nf * 8, nf * 4, kernel_size=3, padding=1, bias=False)),
        nn.LeakyReLU(0.2, False))
        self.conv3 = nn.Sequential(
        norm_layer(nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1, bias=False)),
        nn.LeakyReLU(0.2, False))
        self.conv4 = nn.Sequential(
        norm_layer(nn.Conv2d(nf * 2, nf * 1, kernel_size=3, padding=1, bias=False)))

        self.ffm1=FF_Module(nf * 8)
        self.ffm2 = FF_Module(nf * 4)
        self.ffm3 = FF_Module(nf * 2)
        self.adain1 = AdainBlock(dim=nf * 4, dimin=nf)
        self.adain2 = AdainBlock(dim=nf * 2, dimin=nf)
        self.adain3 = AdainBlock(dim=nf* 1, dimin=nf)
        self.mapping1 = MappingNetwork(nf * 4, nf * 2, nf)
        self.mapping2 = MappingNetwork(nf * 2, nf , nf)
        self.mapping3 = MappingNetwork(nf, nf, nf)
        self.lrelu= nn.LeakyReLU(0.2, False)
        '''373'''
        # self.prelu = nn.PReLU()
        self.mapping = MappingNetwork(nf * 8, nf * 4, nf)


        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

        if self.opt.isTrain:
            self.queue=torch.zeros((0,64),dtype=torch.float).cuda()



    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    def forward(self, ref_img, real_img, seg_map, ref_seg_map):
        coor_out = {}
        batch_size = ref_img.shape[0]
        image_height = ref_img.shape[2]
        image_width = ref_img.shape[3]

        seg_input = seg_map
        adaptive_feature_seg = self.adaptive_model_seg(seg_input, seg_input)
        adaptive_feature_img = self.adaptive_model_img(ref_img, ref_img)
        for i in range(len(adaptive_feature_seg)):
            adaptive_feature_seg[i] = util.feature_normalize(adaptive_feature_seg[i])
            adaptive_feature_img[i] = util.feature_normalize(adaptive_feature_img[i])
        if self.opt.isTrain and self.opt.weight_novgg_featpair > 0:
            adaptive_feature_img_pair = self.adaptive_model_img(real_img, real_img)
            loss_novgg_featpair = 0
            weights = [1.0, 1.0, 1.0, 1.0]
            for i in range(len(adaptive_feature_img_pair)):
                adaptive_feature_img_pair[i] = util.feature_normalize(adaptive_feature_img_pair[i])
                loss_novgg_featpair += F.l1_loss(adaptive_feature_seg[i], adaptive_feature_img_pair[i]) * weights[i]
            coor_out['loss_novgg_featpair'] = loss_novgg_featpair * self.opt.weight_novgg_featpair

            # if self.opt.mcl:
            #     feat_k, sample_ids = self.patch_sample(adaptive_feature_seg[0], 64, None)
            #     feat_q, _ = self.patch_sample(adaptive_feature_img_pair[0], 64, sample_ids)
            #     nceloss = self.nceloss(feat_k, feat_q)
            #     coor_out['nceloss'] = nceloss * self.opt.nce_w
            #
            # 对应位置 信息 需要改进
            if self.opt.mcl:
                total_nceloss=0
                total_nceloss_img = 0
                n=len(adaptive_feature_seg)
                for i in range(n):
                    feat_k, sample_ids = self.patch_sample(adaptive_feature_seg[i], 64, None) #CUT输入图像，z+,z-
                    feat_q, _ = self.patch_sample(adaptive_feature_img_pair[i], 64, sample_ids) #CUT输出图像,z
                    ##############################lossMLC
                    # nceloss = self.nceloss(feat_k, feat_q)* self.opt.nce_w
                    nceloss = self.nceloss(feat_q, feat_k)* self.opt.nce_w
                    total_nceloss += nceloss.mean()
                nceloss_seg=total_nceloss/n
                m=len(adaptive_feature_img)
                for i in range(m):
                    feat_k, sample_ids = self.patch_sample(adaptive_feature_img[i], 64, None)
                    feat_q, _ = self.patch_sample(adaptive_feature_img[i], 64, sample_ids)
                    nceloss_img = self.nceloss(feat_q, feat_k)* self.opt.nce_w
                    total_nceloss_img += nceloss_img.mean()
                nceloss_img=total_nceloss_img /m
                coor_out['nceloss']=(nceloss_seg +nceloss_img)/2


        cont_features=adaptive_feature_seg[0]
        ref_features=adaptive_feature_img[0]
        pos = self.pos_embed

        x4=cont_features

        for i in range(self.recurrence):
            x4=self.cc_attn1(x4,ref_features,ref_features,pos)


        # x5 = self.layer5(x4)  # b 512 16 16
        # # bottleneck = self.bottleneck(self.actvn(x5))  # b 512 16 16
        # bottleneck=self.layer(x5) # b 512 16 16
        # up0 = self.up_0(torch.cat((bottleneck, x5), dim=1),seg_input)  # b 1024 -512
        # up1 = self.up_1(torch.cat((self.up(up0), x4), dim=1),seg_input)  #1024 32 32->512,32,32
        # up1=self.conv1(up1)
        # up2 = self.up_2(torch.cat((self.up(up1), adaptive_feature_seg[1]), dim=1),seg_input)  # 256 64 64
        # up2 = self.conv2(up2)
        # up3 = self.up_3(torch.cat((self.up(up2), adaptive_feature_seg[2]), dim=1),seg_input)  # 128 128 128
        # up3 = self.conv3(up3)
        # up4 = self.up_4(torch.cat((self.up(up3), adaptive_feature_seg[3]), dim=1),seg_input)  # 64 256 256
        # up4 = self.conv4(up4)


        x5 = self.layer5(x4)
        bottleneck = self.layer(x5)
        up0 = self.conv0(torch.cat((bottleneck, x5), dim=1))
        up0 = self.up_0(self.up(up0), seg_input)
        up1 = self.conv1(torch.cat((up0, x4), dim=1))  # 1024 32 32->512,32,32
        up1 = self.up_1(self.up(up1), seg_input)
        ref1 = self.mapping1(adaptive_feature_img[1])
        x31 = self.ffm1(adaptive_feature_seg[1], adaptive_feature_img[1])
        x3 = self.adain1(x31, ref1)
        x3=self.lrelu(x3)
        up2 = self.conv2(torch.cat((up1, x3), dim=1))  # 256 64 64
        up2 = self.up_2(self.up(up2), seg_input)
        ref2 = self.mapping2(adaptive_feature_img[2])
        x21 = self.ffm2(adaptive_feature_seg[2], adaptive_feature_img[2])
        x2 = self.adain2(x21, ref2)
        x2 = self.lrelu(x2)
        up3 = self.conv3(torch.cat((up2, x2), dim=1))  # 128 128 128
        up3 = self.up_3(self.up(up3), seg_input)
        ref3 = self.mapping3(adaptive_feature_img[3])
        x11 = self.ffm3(adaptive_feature_seg[3], adaptive_feature_img[3])
        x1 = self.adain3(x11, ref3)
        x1 = self.lrelu(x1)
        up4 = torch.cat((up3, x1), dim=1)  # 64 256 256
        up4 = self.up_4(self.conv4(up4), seg_input)

        x = F.leaky_relu(up4, 2e-1)  # 64 256 256
        # x = self.res_block(x)
        x = self.conv_img(x)
        x = torch.tanh(x)
        coor_out['fake_image'] = x


        #　training, 100 epoch start 第 100轮的风格当作 负样例
        if self.opt.isTrain and self.opt.contrastive_weight > 0. and self.opt.epoch > 100:
            ref_mapping=self.mapping(ref_features)
            if self.opt.epoch <= 101:
                ref_mapping = ref_mapping.detach()
                self.queue = queue_data(self.queue, ref_mapping)
                self.queue = dequeue_data(self.queue, K=1024)
            else:
                ref_mapping = ref_mapping.detach()
                fake_features = self.adaptive_model_img(x,x)
                # 调制一下
                fake_feature = util.feature_normalize(fake_features[0])
                fake_mapping = self.mapping(fake_feature) # z
                # ref_mapping z+
                # z- 用个queue存 负样例, 将其他refer 风格当作负样例
                contrastive_loss = calc_contrastive_loss(fake_mapping, ref_mapping, self.queue)
                coor_out['contrastive_loss'] = contrastive_loss * self.opt.contrastive_weight
                self.queue = queue_data(self.queue, ref_mapping)
                self.queue = dequeue_data(self.queue, K=1024)

        return coor_out
