import torch
from torch import nn

class PKConv(nn.Module):
    def __init__(self, features, M: int = 3, G= 5, r= 2.5, stride: int = 1, L: int = 32):
        super().__init__()
        d = int(features / r )# 2
        # d = int(3)
        self.M = M
        self.features = features

        self.kernel_size = 1
        padding = self.kernel_size // 2
        self.conv = nn.Conv2d(2, 1, self.kernel_size, 1, padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feats): # feats: concat(x1,x2,x3)
        feats_origin = feats # 1024,5,20,20
        batch_size = feats.shape[0] # 1024

        max_pool_out, _ = torch.max(feats, dim=1, keepdim=True)
        avg_pool_out = torch.mean(feats, dim=1, keepdim=True)
        pool_out = torch.cat([max_pool_out, avg_pool_out], dim=1)

        out = self.conv(pool_out)
        out = self.sigmoid(out)
        feats =  out * feats

        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])  # 1024,3,15,20,20
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        feats_final = torch.relu(feats_Z)
        # 3.select
        attention_vectors = [fc(feats_final) for fc in self.fcs]  # list[0]:1024,5,1,1   list[1]:1024,5,1,1  list[2]:1024,5,1,1
        attention_vectors = torch.cat(attention_vectors, dim=1) # 1024,15,1,1  H
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1) # 1024,3,5,1,1
        attention_vectors = self.softmax(attention_vectors) # 1024,3,5,1,1  w1w2w3:1024,5,1,1
        feats_V = torch.sum(feats * attention_vectors, dim=1) # 1024,3,5,20,20 * 1024,3,5,1,1 ---> 1024,5,20,20
        return feats_V # channel=5