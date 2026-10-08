import torch
import numpy as np
import os
import math
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn import functional as F
from PK_Attention import PKConv
from SE_Module import SELayer
from Attention_Module import MultiHeadAttention

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class PyramidE(torch.nn.Module):
    def __init__(self, logger, num_emb, embedding_dim=200, input_drop=0.5, hidden_drop=0.5, feature_map_drop = 0.5,k_w = 10, k_h = 20,
                 output_channel = 5, filter1_size = (1,3), filter2_size = (3,3), filter3_size = (1,5),filter4_size = (1,3), filter5_size = (3,3), filter6_size = (1,5),batch_size=1024):
        super(PyramidE, self).__init__()
        current_file_name = os.path.basename(__file__)
        logger.info( "[Model Name]: " + str(current_file_name))
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.emb = torch.nn.Embedding(num_emb, embedding_dim)
        self.logger = logger
        self.kernel_outchannel = [5, 5, 5]
        self.se1 = SELayer(self.kernel_outchannel[0], ratio = int(0.5*output_channel))
        self.se3 = SELayer(self.kernel_outchannel[1], ratio = int(0.5*output_channel))
        self.se5 = SELayer(self.kernel_outchannel[2], ratio = int(0.5*output_channel))
        self.SK = PKConv(self.kernel_outchannel[0])
        n_head = 8
        print("n_head:",n_head)
        # self.att = MultiHeadAttention(self.kernel_outchannel[0]+ self.kernel_outchannel[1] + self.kernel_outchannel[2], 1)
        # self.att = MultiHeadAttention(n_head, self.kernel_outchannel[0] + self.kernel_outchannel[1] + self.kernel_outchannel[2]+1,64,100)
        # self.att_0 = MultiHeadAttention(n_head, 1 ,64,64)
        # self.att   = MultiHeadAttention(n_head, self.kernel_outchannel[0] ,64,64)
        # self.att_1 = MultiHeadAttention(n_head, self.kernel_outchannel[0] ,32,50)
        # self.att_1 = MultiHeadAttention(n_head, self.kernel_outchannel[0],64,64)
        # self.att_11 = MultiHeadAttention(n_head, 32,64,64)
        # self.att_2 = MultiHeadAttention(self.kernel_outchannel[1], 1)
        # self.att_3 = MultiHeadAttention(n_head, self.kernel_outchannel[2], 64,64)
        # self.att_31 = MultiHeadAttention(n_head, 32, 64,64)
        self.att =  MultiHeadAttention(n_head, 400 ,32 ,50)

        self.embedding_dim = embedding_dim
        self.perm = 1
        self.k_w = k_w
        self.k_h = k_h
        self.loss = torch.nn.CrossEntropyLoss()
        self.device = torch.device('cuda')
        self.chequer_perm1 = self.get_chequer_perm1()
        self.reshape_H = 20
        self.reshape_W = 20
        self.in_channel = 1

        # Link
        # self.kernel_inchannel = 1
        self.kernel_inchannel1 = [1, 2, 8]
        print("self.kernel_inchannel1:", self.kernel_inchannel1)
        # self.kernel_inchannel2 = [1, 2, 4] # [2,4,4] 6不行，1024
        self.emb = torch.nn.Embedding(num_emb, embedding_dim)
        self.filter1_size = filter1_size
        self.filter2_size = filter2_size
        self.filter3_size = filter3_size
        self.filter4_size = filter4_size
        self.filter5_size = filter5_size
        self.filter6_size = filter6_size
        self.point_conv = torch.nn.Conv2d(5, 5, kernel_size=1, stride=1)
        # self.point_conv_1 = torch.nn.Conv2d(self.kernel_outchannel[0] + self.kernel_outchannel[1] + self.kernel_outchannel[2], self.kernel_outchannel[0] + self.kernel_outchannel[1] + self.kernel_outchannel[2], kernel_size=1, stride=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # filter_dim = self.kernel_inchannel * self.kernel_outchannel * self.filter2_size[0] * self.filter2_size[1]  # in*out*h*w
        # self.filter = torch.nn.Embedding(self.num_emb, filter_dim, padding_idx=0).to(device)    # 1024,45,0

        filter1_dim = self.kernel_inchannel1[0] * self.kernel_outchannel[0] * self.filter1_size[0] * self.filter1_size[1]  # in*out*h*w
        self.filter1 = torch.nn.Embedding(self.num_emb, filter1_dim, padding_idx=0).to(self.device)  # 41817,15,0
        filter2_dim = self.kernel_inchannel1[1] * self.kernel_outchannel[1] * self.filter2_size[0] * self.filter2_size[1]  # 2*5*3*3=90
        self.filter2 = torch.nn.Embedding(self.num_emb, filter2_dim, padding_idx=0).to(self.device)  # 41817,90,0
        filter3_dim = self.kernel_inchannel1[2] * self.kernel_outchannel[2] * self.filter3_size[0] * self.filter3_size[1]  # 8*5*1*5=200
        self.filter3 = torch.nn.Embedding(self.num_emb, filter3_dim, padding_idx=0).to(self.device)  # 41817,200,0

        self.input_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feature_map_drop)
        self.bn0 = torch.nn.BatchNorm2d(self.in_channel)
        self.bn1 = torch.nn.BatchNorm2d(self.kernel_outchannel[0] + self.kernel_outchannel[1] + self.kernel_outchannel[2])
        self.bn1_1 = torch.nn.BatchNorm2d(self.kernel_outchannel[0])
        self.bn1_2 = torch.nn.BatchNorm2d(self.kernel_outchannel[1])
        self.bn1_3 = torch.nn.BatchNorm2d(self.kernel_outchannel[2])
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim) # 200

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        fc_length = self.reshape_H * self.reshape_W * (self.kernel_outchannel[0]*6+1)
        self.fc = torch.nn.Linear(fc_length, embedding_dim)
        self.b = torch.nn.Parameter(torch.zeros(num_emb))# 1×41817

        print("--->num_emb={}, embedding_dim={}, input_drop={}, hidden_drop={}, feature_map_drop = {},k_w = {}, k_h = {},kernel_outchannel = {}, filter1_size={}, filter2_size={}, filter3_size={},filter4_size ={}, filter5_size ={}, filter6_size={} , fc_length={}".format(
                num_emb, embedding_dim, input_drop, hidden_drop, feature_map_drop, k_w, k_h, self.kernel_outchannel,
                filter1_size, filter2_size, filter3_size, filter4_size, filter5_size, filter6_size, fc_length))

        self.Q = torch.nn.Parameter(torch.randn(200,400))
        self.K = torch.nn.Parameter(torch.randn(200,400))
        self.V = torch.nn.Parameter(torch.randn(200,self.kernel_outchannel[0] + self.kernel_outchannel[1] + self.kernel_outchannel[2]))
        # self.V = torch.nn.Parameter(torch.randn(200,self.kernel_outchannel[0]))
        # self.V = torch.nn.Parameter(torch.randn(200,200))

        self.Q1 = torch.nn.Parameter(torch.randn(200, 200))
        self.K1 = torch.nn.Parameter(torch.randn(200, 200))
        self.V1 = torch.nn.Parameter(torch.randn(200,self.kernel_outchannel[0] + self.kernel_outchannel[1] + self.kernel_outchannel[2]))
        # self.V1 = torch.nn.Parameter(torch.randn(200, self.kernel_outchannel[0]))

    def to_var(self, x, use_gpu=True):
        if use_gpu:
            return Variable(torch.from_numpy(x).long().cuda())

    def init(self):
        torch.nn.init.kaiming_normal(self.emb.weight.data)
        torch.nn.init.kaiming_normal(self.filter1.weight.data)
        torch.nn.init.kaiming_normal(self.filter2.weight.data)
        torch.nn.init.kaiming_normal(self.filter3.weight.data)
        # torch.nn.init.xavier_normal_(self.filter4.weight.data)
        # torch.nn.init.xavier_normal_(self.filter5.weight.data)
        # torch.nn.init.xavier_normal_(self.filter6.weight.data)
        torch.nn.init.kaiming_normal(self.point_conv.weight.data)
        torch.nn.init.kaiming_normal(self.fc.weight.data)

    def get_chequer_perm1(self):
        ent_perm = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)]) #(1,200)
        rel_perm = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)])
        comb_idx = []
        for k in range(self.perm):
            temp = []
            ent_idx, rel_idx = 0, 0
            for i in range(self.k_h): # 20
                for j in range(self.k_w): # 10
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1
            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm # (1,400)

    def forward(self, e1, rel):
        # print("e1_1_size:", e1)
        e1 = self.to_var(e1)
        rel = self.to_var(rel)
        e1_embedded = self.emb(e1) #1024,200
        # print("e1_embedded：",e1_embedded.size()) # 948
        rel_embedded = self.emb(rel)

        Q = torch.mm(e1_embedded, self.Q) # 1024,200
        K = torch.mm(rel_embedded, self.K) # 1024,200
        V = torch.mm(rel_embedded, self.V) # 1024,15
        res = torch.mm(Q, K.T) / np.sqrt(200) # 1024,1024
        res = torch.softmax(res, dim=1)
        atten_rel = torch.mm(res, V) # 1024,15

        Q1 = torch.mm(rel_embedded, self.Q1)  # 1024,200
        K1 = torch.mm(e1_embedded, self.K1)  # 1024,200
        V1 = torch.mm(e1_embedded, self.V1)  # 1024,15
        res = torch.mm(Q1, K1.T) / np.sqrt(200)  # 1024,1024
        res = torch.softmax(res, dim=1)
        atten_e1 = torch.mm(res, V1)  # 1024,15

        # Chequer
        comb_emb = torch.cat([e1_embedded, rel_embedded], dim=1) # 1024,400
        chequer_perm1 = comb_emb[:, self.chequer_perm1] # 1024,1,400
        stack_inp1 = chequer_perm1.reshape((-1, self.perm, 2*self.k_w, self.k_h)) # 2*self.k_w  (1024,1,20,20)
        x = self.bn0(stack_inp1) # 1024,1,20,20
        x = self.input_drop(x)
        x_origin = x
        # x_normal = x  # 普通卷积使用 x_origin：(N, C, H, W) -->1024,1,20,20
        # Self-att 不行
        # x = x.view(e1_embedded.size(0), self.reshape_H * self.reshape_W, -1)  # 1024,400,1
        # x = self.att_0(x, x, x)
        # x = x.view(e1_embedded.size(0), -1, self.reshape_H, self.reshape_W) # 1024,15,20,20

        x_1 = x.view(self.batch_size,1,-1) # 1024,1,400
        x_1 = self.att(x_1, x_1, x_1)
        x_1 = x_1.view(e1_embedded.size(0), -1, self.reshape_H, self.reshape_W)


        x = x.permute(1, 0, 2, 3)
        # f = self.filter(rel) # 1024,9
        # f = f.reshape(e1_embedded.size(0) * self.in_channel * self.kernel_outchannel, -1, self.filter2_size[0],self.filter2_size[1]) # 1024,1,3,3
        f1 = self.filter1(rel)  # (1024,15)
        f1 = f1.reshape(e1_embedded.size(0) * self.in_channel * self.kernel_outchannel[0], -1, self.filter1_size[0],self.filter1_size[1])  # (1024*1*5,1,1,3)-(5120,1,1,3)
        f2 = self.filter2(rel)  # (1024,90)
        f2 = f2.reshape(e1_embedded.size(0) * self.in_channel * self.kernel_outchannel[1], -1, self.filter2_size[0],self.filter2_size[1])
        f3 = self.filter3(rel)  # (1024,200)
        f3 = f3.reshape(e1_embedded.size(0) * self.in_channel * self.kernel_outchannel[2], -1, self.filter3_size[0],self.filter3_size[1])

        # normal_conv
        # x_normal = F.conv2d(x, f, padding=(int((self.filter2_size[0] - 1) // 2),int((self.filter2_size[1] - 1) // 2)))
        # x_normal = x_normal.permute(1, 0, 2, 3)
        # Ablation
        x1 = F.conv2d(x, f1, groups=e1_embedded.size(0) // self.kernel_inchannel1[0], dilation=1,padding=(int((self.filter1_size[0] - 1) // 2),int((self.filter1_size[1] - 1) // 2)))
        # x1:1,1024,20,20    f1:(5120,1,1,3)    out:1,5120,20,20
        x1 = x1.reshape(e1_embedded.size(0), self.kernel_outchannel[0], self.reshape_H, self.reshape_W)  # NCHW (1024,5,20,20)
        x1 = self.bn1_1(x1)
        # x1 = F.relu(x1)
        x1 = self.leaky_relu(x1)
        # x1 = self.se1(x1) # 1024,5,20,20
        # # x1 = self.avgpool(x1)
        # x1 = self.point_conv(x1)
        # x1 = self.bn1_1(x1)
        # x1 = self.leaky_relu(x1)

        # Self-att
        # x1 = x1.view(e1_embedded.size(0), self.reshape_H * self.reshape_W, -1)  # 1024,400,5
        # x1 = self.att_1(x1, x1, x1)
        # x1 = x1.view(e1_embedded.size(0), -1, self.reshape_H, self.reshape_W)
        # x1 = self.bn1_1(x1)
        # x1 = self.leaky_relu(x1)

        # x2 = F.conv2d(x, f2, groups=e1_embedded.size(0) // self.kernel_inchannel1[1], dilation=1, padding=(int((self.filter2_size[0] - 1) // 2),int((self.filter2_size[1] - 1) // 2)))  # NCHW(1,1024,20,20) OIHW(5120,2,3,3)   pad=(1,1)
        # x2 = x2.reshape(e1_embedded.size(0), self.kernel_outchannel[1], self.reshape_H,self.reshape_W)  # (1024,5,20,20)
        # x2 = self.bn1_2(x2)
        # # x21 = F.relu(x21)
        # x2 = self.leaky_relu(x2)
        #
        # x2 = self.point_conv(x2)
        # x2 = self.bn1_1(x2)
        # # x1 = F.relu(x1)
        # x2 = self.leaky_relu(x2)
        #
        # x2 = self.se3(x2)
        # x2 = x2.view(e1_embedded.size(0), self.reshape_H * self.reshape_W, -1)
        # x2 = self.att_2(x2, x2, x2)
        # x2 = x2.view(e1_embedded.size(0), -1, self.reshape_H, self.reshape_W)
        # Ablation
        x21 = F.conv2d(x, f2, groups=e1_embedded.size(0) // self.kernel_inchannel1[1], dilation=1,padding=(int((self.filter2_size[0] - 1) // 2),int((self.filter2_size[1] - 1) // 2)))  # NCHW(1,1024,20,20) OIHW(5120,2,3,3)   pad=(1,1)
        x21 = x21.reshape(e1_embedded.size(0), self.kernel_outchannel[1], self.reshape_H, self.reshape_W)  # (1024,5,20,20)
        x21 = self.bn1_2(x21)
        # x21 = F.relu(x21)
        x21 = self.leaky_relu(x21)

        x22 = F.conv2d(x, f2, groups=e1_embedded.size(0) // self.kernel_inchannel1[1], dilation=2,padding=(int((5 - 1) // 2), int((5 - 1) // 2)))  # NCHW(1,1024,20,20) OIHW(5120,2,3,3)   pad=(1,1)
        x22 = x22.reshape(e1_embedded.size(0), self.kernel_outchannel[1], self.reshape_H, self.reshape_W)  # (1024,5,20,20)
        x22 = self.bn1_2(x22)
        # x22 = F.relu(x22)
        x22 = self.leaky_relu(x22)

        # x23 = F.conv2d(x, f2, groups=e1_embedded.size(0) // self.kernel_inchannel1[1], dilation=4, padding=(int((9 - 1) // 2), int((9 - 1) // 2)))  # NCHW(1,1024,20,20) OIHW(5120,2,3,3)   pad=(1,1)
        x23 = F.conv2d(x, f2, groups=e1_embedded.size(0) // self.kernel_inchannel1[1], dilation=3, padding=(int((7 - 1) // 2), int((7 - 1) // 2)))  # NCHW(1,1024,20,20) OIHW(5120,2,3,3)   pad=(1,1)
        x23 = x23.reshape(e1_embedded.size(0), self.kernel_outchannel[1], self.reshape_H,self.reshape_W)  # (1024,5,20,20)
        x23 = self.bn1_2(x23)
        # x23 = F.relu(x23)
        x23 = self.leaky_relu(x23)
        # SK  (☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆ Ablation-2)
        x2 = torch.cat([x21, x22, x23], dim=1)
        x_2 = x2
        x2 = self.SK(x2) # 1024,15,20,20

        # # Self-Attention
        # x2 = x2.view(e1_embedded.size(0), self.reshape_H * self.reshape_W, -1)  # 1024,400,15
        # x2 = self.att(x2, x2, x2)
        # x2 = x2.view(e1_embedded.size(0), -1, self.reshape_H, self.reshape_W)  # 1024,15,20,20

        x3 = F.conv2d(x, f3, groups=e1_embedded.size(0) // self.kernel_inchannel1[2],dilation=1, padding=(int((self.filter3_size[0] - 1) // 2),int((self.filter3_size[1] - 1) // 2)))  # # NCHW(1,1024,20,20) OIHW(5120,4,1,5)   pad=(0,2)
        # x3 = self.point_conv(x3)
        x3 = x3.reshape(e1_embedded.size(0), self.kernel_outchannel[2], self.reshape_H, self.reshape_W)  # (128,5,20,20)
        x3 = self.bn1_3(x3)
        # x3 = F.relu(x3)
        x3 = self.leaky_relu(x3) # 1024,5,20，20
        # x3 = self.se5(x3)
        # x3 = self.point_conv(x3)
        # x3 = self.bn1_3(x3)
        # x3 = self.leaky_relu(x3)
        # Self-att
        # x3 = x3.view(e1_embedded.size(0), self.reshape_H * self.reshape_W, -1)
        # x3 = self.att_3(x3, x3, x3)
        # x3 = x3.view(e1_embedded.size(0), -1, self.reshape_H, self.reshape_W)
        # x3 = self.bn1_1(x3)
        # x3 = self.leaky_relu(x3)

        # x = x1 + x2 + x3
        # y1, y2, y3 = self.att(x)
        # y1 = y1.expand_as(x1)
        # y2 = y3.expand_as(x2)
        # y3 = y1.expand_as(x3)
        # x1 = x1 * y1
        # x2 = x2 * y2
        # x3 = x3 * y3

        x = torch.cat([x_1, x1, x2, x_2, x3], dim=1)  #1,5,5,15,5 1024,17,20,20
        # x = torch.cat([x_1, x1, x2, x3, x_origin], dim=1)  # 1024,17,20,20
        # x = torch.cat([x_1, x_origin], dim=1)  # 1024,17,20,20
        # x = torch.cat([x1, x21,x22,x23, x3], dim=1) # 1024,25,20,20
        x = self.leaky_relu(x)
        x = self.feature_map_drop(x)
        # Ablation
        # atten = atten_e1 * atten_rel
        # atten = atten_rel # 1024,15
        # x = atten.view(self.batch_size, self.kernel_outchannel[0] + self.kernel_outchannel[1] + self.kernel_outchannel[2], 1, 1) * x #1024,15,
        # x = atten.view(self.batch_size, self.kernel_outchannel[0], 1, 1) * x  # 1024,,
        # x = x.sum(dim=1)

        # Self-att & res
        # x_before_att = x
        # x = x.view(e1_embedded.size(0), self.reshape_H * self.reshape_W, -1)  # 1024,400,15
        # x = self.att(x, x, x)
        # x = x.view(e1_embedded.size(0), -1, self.reshape_H, self.reshape_W) # 1024,15,20,20
        # x = torch.cat([x, x_before_att], dim=1)  # 1024,30,20,20

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        weight = self.emb.weight
        weight = weight.transpose(1, 0)
        x = torch.mm(x, weight) # 1024,200 *200,41817 -> 1024,41817
        x += self.b.expand_as(x)
        pred = x

        return pred
