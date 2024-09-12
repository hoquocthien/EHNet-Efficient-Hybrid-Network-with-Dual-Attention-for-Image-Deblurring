import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import einops


class MHTA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(MHTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_dwconv = nn.Conv2d(dim, dim*3, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(x)

        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


if __name__ == '__main__':
##  Test Spatial Feature    
    num_heads = 8   
    batch_size = 8
    num_channels = 48
    height = 64
    weight = 64
    dim = height*weight
    x = torch.randn(batch_size, num_channels, height, weight)
    mask = torch.ones([1, num_channels], dtype=torch.bool)
    attn = MHTA(num_channels, shifts=4, window_sizes=[4,8,16])
    # print(attn)
    y = attn(x)
    print(y.shape)
    


        



