'''
This code is borrowed from Serge-weihao/CCNet-Pure-Pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import util.util as util


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


"""构建一个层归一化（layernorm）模块"""
class LayerNorm(nn.Module):
    # 初始化函数，接收features（特征维度大小）和eps（防止除以零的微小值）作为输入参数
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()  # 调用父类nn.Module的构造函数
        self.gamma = nn.Parameter(torch.ones(features))  # 定义一个大小为features的一维张量，初始化为全1，并将其设置为可训练参数
        self.beta = nn.Parameter(torch.zeros(features))  # 定义一个大小为features的一维张量，初始化为全0，并将其设置为可训练参数
        self.eps = eps  # 将防止除以零的微小值eps保存为类实例的属性

    # 定义前向传播函数，输入参数x是输入张量
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # 计算输入x在最后一个维度上的均值，保持输出结果的维度
        std = x.std(-1, keepdim=True)  # 计算输入x在最后一个维度上的标准差，保持输出结果的维度
        # 对输入x进行层归一化，使用可训练参数a_2和b_2进行缩放和偏移，最后返回归一化后的结果
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def PositionalNorm2d(x, epsilon=1e-5):
    # x: B*C*W*H normalize in C dim
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output


class CrissCrossAttention1(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention1, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(normalized_shape=[512, 32, 32], elementwise_affine=False)

    def forward(self, q, k, v, pos):
        pos = pos.repeat(q.shape[0], 1, 1, 1)
        q_pos = q + pos
        k_pos = k + pos
        m_batchsize, _, height, width = q_pos.size()  # torch.Size([2, 258, 64, 64])
        # pos = pos.repeat(q.shape[0], 1, 1, 1)
        # q = torch.cat([util.feature_normalize(q), pos], dim=1)
        # k = torch.cat([util.feature_normalize(k), pos], dim=1)
        proj_query = self.query_conv(q_pos)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(k_pos)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(v)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size()) # torch.Size([2, 512, 32, 32])
        # return self.gamma*(out_H + out_W) + q
        # return PositionalNorm2d(self.gamma*(out_H + out_W) + q )
        return self.norm(self.gamma * (out_H + out_W) + q)



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CrissCrossAttention1(64)
    model = model.to(device)
    x = torch.randn(2, 64, 5, 6)
    out = model(x.to(device))
    print(out.shape)
