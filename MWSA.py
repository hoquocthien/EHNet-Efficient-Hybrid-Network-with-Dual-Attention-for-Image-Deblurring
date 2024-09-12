import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import einops
from ptflops import get_model_complexity_info
import time

class MWSA(nn.Module):
    def __init__(self, channels, shifts=4, window_sizes=[4, 8, 16],bias=False):
        super(MWSA, self).__init__()    
        
        self.channels = channels
        self.shifts   = shifts
        self.window_sizes = window_sizes    
        self.dw = nn.Conv2d(channels, channels*3, kernel_size=3, groups= channels, padding=1, bias=bias)
        self.out = nn.Conv2d(channels*3//2, channels, kernel_size=1, bias=bias)
        self.split_chns  = [channels, channels, channels]

        
    def forward(self, x):
        
        b,c,h,w = x.shape
        x = self.dw(x)
        ys = []
        xs = torch.split(x, self.split_chns, dim=1)
        
        for idx, x_ in enumerate(xs):
            
            wsize = self.window_sizes[idx]
            if self.shifts > 0:
                x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,3))
            
            q, v = rearrange(
                x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c', 
                qv=2, dh=wsize, dw=wsize
            )
                        
            atn = (q @ q.transpose(-2, -1)) 
            
            atn = atn.softmax(dim=-1)
            y_ = (atn @ v)
            y_ = rearrange(
                y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', 
                h=h//wsize, w=w//wsize, dh=wsize, dw=wsize
            )
            ###
            
            if self.shifts > 0:
                y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2, 3))
            
            ys.append(y_)
            
        
        y = torch.cat(ys, dim=1)      
        y = self.out(y)
        return y
        

if __name__ == '__main__':
##  Test Spatial Feature    
    batch_size = 2
    num_channels = 48
    height = 256
    weight = 256
    # dim = height*weight
    x = torch.randn(batch_size, num_channels, height, weight)
    mask = torch.ones([1, num_channels], dtype=torch.bool)
    attn = MWSA(num_channels, shifts=4, window_sizes=[4,8,16])

    model = attn
    inp_shape = (num_channels, height,weight)
    mac, params = get_model_complexity_info(model, inp_shape, verbose= False, print_per_layer_stat=False)
    # print(attn)
    print(mac,params)
    # y = attn(x)
    # print(y.shape)
    


        



