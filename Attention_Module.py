import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, **kwargs): # d_model is token_dim  # 2,5,64,64
        super().__init__()
        self.reshape_H = 20
        self.reshape_W = 20
        self.n_head = n_head # 32
        self.d_k = d_k # 64
        self.d_v = d_v  # 64
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                   dropout=0.2)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm_in = nn.LayerNorm((3, n_head * d_v))
        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(0.3)
        self.pos_ffn = PositionwiseFeedForwardUseConv(d_model, 512, dropout=0.2)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # print("q:",q.size())
        batch_size, len_q, channels = q.size() # batch_size,seq_length,token_dim 1024,400,5
        batch_size, len_k, channels = k.size()
        batch_size, len_v, channels = v.size()
        residual = q
        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k) # q:1024,400,5 *  w_q:5,64 ->1024,400,64  -> 1024,400,2,32
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v) # 1024,400,2,50
        q = q.transpose(1, 2).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk 1024,2,400,32 ->2048,400,32
        k = k.transpose(1, 2).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.transpose(1, 2).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv 2048,400,50
        output = self.attention(q, k, v)  # 2048,400,400 * 2048,400,50 -> 2024,400,50
        output = output.view(batch_size, n_head, len_q, d_v) # 1024,2,400,50
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)  # b x lq x (n*dv) 1024,400,100
        output = self.dropout(self.fc(output)) #  1024,400,100 * 100,5 ->1024,400,5
        output = self.layer_norm(output) # 1024,1,400
        output = self.pos_ffn(output) #
        return output

class PositionwiseFeedForwardUseConv(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.3,): #400,512
        super(PositionwiseFeedForwardUseConv, self).__init__()
        self.fc1 = nn.Linear(1, d_hid) # 1024,
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,padding=1,
                                groups=512)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=5,padding=2,
                                groups=512)
        self.fc2 = nn.Linear(512, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):# 1024,1,400
        # residual = x
        batch_size,_,_ = x.size()
        x = x.transpose(2,1) # 1024,400,1
        x = self.fc1(x)# 1024,400,512
        x = x.transpose(2,1).contiguous().view(batch_size,512,20,20)
        # x1 = F.gelu(self.conv1(x))
        # x2 = F.gelu(self.conv2(x)) # 1024,512,20,20
        x1 = self.conv1(x)
        x2 = self.conv2(x) # 1024,512,20,20
        x = (x1 + x2).view(batch_size,512,400).transpose(2,1) # 1024,400,512
        x = self.fc2(x).transpose(2,1) # 1024,400,1 -> 1024,1,400
        # x = self.drop(x)
        x = self.layer_norm(x)
        return x

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2)) # 1024,200,200
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output
# Ablation: \d-ffn
# class PositionwiseFeedForwardUseConv(nn.Module):
#     def __init__(self, d_in, d_hid, dropout=0.3,):
#         super(PositionwiseFeedForwardUseConv, self).__init__()
#         self.w_1 = nn.Conv1d(d_in, d_hid, 1)
#         nn.init.kaiming_uniform_(self.w_1.weight)
#         self.w_2 = nn.Conv1d(d_hid, 400, 1)
#         nn.init.kaiming_uniform_(self.w_2.weight)
#         self.layer_norm = nn.LayerNorm(d_in)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         residual = x
#         output = x.transpose(1, 2) # 1024,400,1
#         output = self.w_1(output) # 1024,512,1
#         output = self.w_2(output) # 1024,400,1
#         output = output.transpose(1, 2)
#         output = self.layer_norm(output)
#         return output

# class MultiHeadAttention(nn.Module):
#     def __init__(self, channels, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         # Way 1: channels // num_heads
#         self.head_dim = channels // num_heads  # 15//3 =5   50//3
#         assert self.head_dim * num_heads == channels, "channels must be divisible by num_heads"
#         # Way 2: num_heads * dk
#         self.query_conv = nn.Linear(channels, 50)
#         self.key_conv = nn.Linear(channels, 50)
#         self.value_conv = nn.Linear(channels, channels) # out_channel=5
#         self.softmax = nn.Softmax(dim=-1)  # Apply softmax on the attention scores
#
#     def forward(self, x):
#         # x : [batch_size, channels, height, width] = [1024, 5, 20, 20]
#         batch_size, channels, height, width = x.size()
#         num_patches = height * width  # Flatten spatial dimensions
#         # Flatten the height and width dimensions into one (num_patches)
#         x = x.view(batch_size, channels, num_patches).permute(0, 2, 1)  # Shape: [batch_size, num_patches, channels] 1024,400,5
#
#         # Linear projections for Q, K, V
#         Q = self.query_conv(x)  # Shape: [batch_size, num_patches, channels] 1024,400,5 * 5,64 -》1024,400,64
#         K = self.key_conv(x)    # Shape: [batch_size, num_patches, channels]
#         V = self.value_conv(x)  # Shape: [batch_size, num_patches, channels] 1024,400,5 *
#
#         Q = Q.view(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # 1024,3,400,5
#         K = K.view(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         V = V.view(batch_size, num_patches, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         # Scaled dot-product attention
#         attention_scores = torch.bmm(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(channels, dtype=torch.float32)) # 1024,400,400
#         attention_probs = self.softmax(attention_scores)  # Shape: [batch_size, num_patches, num_patches]
#
#         # Apply attention weights to the values
#         attention_out = torch.bmm(attention_probs, V)  # Shape: [batch_size, num_patches, channels] 1024,400,400 * 1024,400,5 -》1024,400,5
#
#         # Reshape attention output back to [batch_size, channels, height, width]
#         attention_out = attention_out.permute(0, 2, 1).view(batch_size, channels, height, width) # 1024,5,20,20
#
#         return attention_out
#
# # Example usage
# x = torch.randn(1024, 5, 20, 20)  # Input feature map of shape [batch_size, channels, height, width]
# self_attention_layer = MultiHeadAttention(channels=5, num_heads=1)
# output = self_attention_layer(x)
# print(output.shape)  # Output shape should be [1024, 5, 20, 20]

# entity/rel attention
# class MultiHeadAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#
#         self.query_proj = nn.Linear(embed_dim, embed_dim) # 200,200
#         self.key_proj = nn.Linear(embed_dim, embed_dim)
#         self.value_proj = nn.Linear(embed_dim, embed_dim)
#
#         self.scaling_factor = self.head_dim ** -0.5
#
#     def forward(self, query, key, value):
#         batch_size = query.size(0)
#
#         # Project inputs to multi-heads
#         query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
#
#         # Attention mechanism
#         attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling_factor
#         attention_probs = F.softmax(attention_scores, dim=-1)
#         attention_output = torch.matmul(attention_probs, value)
#
#         # Concatenate heads and put through final linear layer
#         attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
#         output = self.out_proj(attention_output)
#
#         return output
#
# # Example usage
# embed_dim = 100  # Embedding dimension
# num_heads = 5    # Number of attention heads

# model = MultiHeadAttention(embed_dim, num_heads)

# batch_h = torch.randint(0, 40943, (128,))  # 128 entities
# batch_r = torch.randint(0, 11, (128,))     # 128 relations

# E = model.E(torch.tensor(batch_h))  # Entity embeddings
# R = model.R(torch.tensor(batch_r))  # Relation embeddings

# atten = model(E, R, R)  # Apply multi-head self-attention
#
# print(atten.shape)  # Output shape: [batch_size, sequence_length, embed_dim]

