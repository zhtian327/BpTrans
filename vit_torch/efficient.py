import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ViT(nn.Module):
    def __init__(self, *, sequence_len, num_patches, dim, transformer):
        super().__init__()
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (q p) -> b q p', p = sequence_len),
            nn.Linear(sequence_len, dim),  # 对分割好的图像块进行线性处理（全连接），输入维度为每一个小块的所有像素个数，输出为dim（函数传入的参数）
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置编码，获取一组正态分布的数据用于训练
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 分类令牌，可训练

        self.transformer = transformer
        self.to_latent = nn.Identity()  # 占位操作

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 正则化
            nn.Linear(dim, sequence_len),  # 线性输出
            Rearrange('b q p -> b (q p)', p = sequence_len) #reshape
        )

    def forward(self, data):
        x = self.to_patch_embedding(data)  # 切块操作，shape (b, n, dim)，b为批量，n为切块数目，dim为最终线性操作时输入的神经元个数
        b, n, _ = x.shape  # shape (b, n, 1024)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # 分类令牌，将self.cls_token（形状为1, 1, dim）赋值为shape (b, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # 将分类令牌拼接到输入中，x的shape (b, n+1, 1024)
        x += self.pos_embedding[:, :(n + 1)]  # 进行位置编码，shape (b, n+1, 1024)

        x = self.transformer(x)  # transformer操作

        x = self.to_latent(x)
        return self.mlp_head(x[:,1:n+1])  # 线性输出