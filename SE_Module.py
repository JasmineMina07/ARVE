import torch
from torch import nn
import math

class channel_attention(nn.Module):
    def __init__(self,channel,ratio):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//ratio,False),
            nn.ReLU(),
            nn.Linear(channel//ratio,channel,False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b,c,h,w = x.size()
        max_pool_out = self.max_pool(x).view([b,c])  # reshae的原因是linear的输入是二维
        avg_pool_out = self.avg_pool(x).view([b,c])

        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)

        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b,c,1,1])  #都不一定view,flatten、squeeze都行
        return out*x


class spacial_attention(nn.Module):
    def __init__(self,kernel_size = 7):
        super(spacial_attention, self).__init__()
        padding = kernel_size//2  # 为了保证卷积后的特征图尺寸不变  我是用h-k+2p+1/s算的，但网上有简便方法就是卷积核尺寸整除2
        self.conv = nn.Conv2d(2,1,kernel_size,1,padding,bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b,c,h,w = x.size()
        max_pool_out,_ = torch.max(x,dim=1,keepdim=True) # 输出的是最大值及其索引 ；通道维度保留
        avg_pool_out = torch.mean(x, dim=1, keepdim=True)  # 在通道维度上求平均值
        pool_out = torch.cat([max_pool_out, avg_pool_out], dim=1)  # 在通道维度上拼接

        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out * x
#
# class SELayer(nn.Module):
#     def __init__(self,channel,ratio=16,kernel_size=7):
#         super(SELayer, self).__init__()
#         self.channel_attention = channel_attention(channel,ratio)
#         self.spacial_attention = spacial_attention(kernel_size)
#
#     def forward(self,x):
#         x = self.channel_attention(x)
#         x = self.spacial_attention(x)
#         return x

# ECA
# class SELayer(nn.Module):
#     def __init__(self,channel,gamma = 2,b = 1): # 需要根据通道数自适应地计算卷积核大小
#         super(SELayer, self).__init__()
#
#         kernel_size = int(abs(math.log(channel,2)+b)/gamma)
#         kernel_size = kernel_size if kernel_size % 2 else kernel_size +1 # 如果k是奇数，返回它；如果是偶数，返回k+1  # 5
#         padding = kernel_size//2  # 2
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1,1,kernel_size ,padding = padding, bias = False) # 当做序列模型来看
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self,x):
#         b ,c ,h, w = x.size()
#         avg =self.avg_pool(x).view([b,1,c]) # [2,512,1,1] -> [2,1,512] 因为下一步要进行1d卷积 其输入是[b,in_channel,sequence_length]
#         out = self.conv(avg) # b,1,c [2,1,512]
#         out = self.sigmoid(out).view([b,c,1,1])
#         return out*x


# SE
class SELayer(nn.Module):
    def __init__(self, channel, ratio):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size() # x: 1024,5,20,20
        y = self.avg_pool(x) # (1024,5,1,1)
        y = y.view(b, c) # (1024,5)
        y = self.fc(y) # (1024,5)
        y = y.view(b, c, 1, 1) # (1024,5,1,1)
        y = y.expand_as(x) # (1024,5,20,20)
        return x * y