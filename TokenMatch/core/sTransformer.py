import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs): #x=b*n+1*c
        return self.fn(self.norm(x), **kwargs) #层归一化

class FeedForward(nn.Module): #进行了一次全连接 --> GELU --> 全连接的变换
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), #高斯误差线性单元函数 GELUs其实是dropout、zoneout、ReLU的综合，GELU对于输入乘以一个 [ 0 , 1 ] [0,1] [0,1]组成的mask，而该mask的生成则是依概率随机的依赖于输入。假设输入为 X X X, mask为 m m m，则 m m m服从一个伯努利分布 Φ ( x ) = P ( X < x ) \Phi(x)=P(X<x) Φ(x)=P(X<x)，其中 X X X服从标准正态分布。这么选择是因为神经元的输入趋向于正态分布，这么设定使得当输入 x x x减小的时候，输入会有一个更高的概率被dropout掉，这样的激活变换就会随机依赖于输入了。
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x): #b*(n+1)*c
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64): #这里的heads=8是不是应该改一下呀
        super().__init__()
        inner_dim = dim_head * heads #dim_head=64 heads=2
        self.heads = heads
        self.scale = dim ** -0.5 #dim =c=128

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) #Linear(in_features=128, out_features=384, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x, mask = None): 
        b, n, _, h = *x.shape, self.heads #x=b*(n+1)*c 
        qkv = self.to_qkv(x).chunk(3, dim = -1) #3个b*(n+1)*c 
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv) #3个b*h*(n+1)*c/h

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale #b*h*(n+1)*(n+1);dots相当于attn
        mask_value = -torch.finfo(dots.dtype).max #-3.4028234663852886e+38

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
########################################################
        mask1 = torch.zeros(b, self.heads, n, n, device=x.device, requires_grad=False) #b*h*(n+1)*(n+1)
        
        index = torch.topk(dots, k=int((4*n)/5), dim=-1, largest=True)[1] #b*h*(n+1)*(n+1)/2;从指定维度 dim 上选择前 k 个最大的元素。
        mask1.scatter_(-1, index, 1.) #b*h*(n+1)*(n+1),里面有0和1；将 mask1 中的所有 index 所对应的位置设置为 1.0，其余位置保持为 0.0
        # dots1 = torch.where(mask1 > 0, dots, torch.full_like(dots, float(0))) #b*h*(n+1)*(n+1) 对于 mask1 中大于 0 的位置，attn1 中的值保持不变（即与 attn 中的值相同）;对于 mask1 中不大于 0 的位置，attn1 中的值被替换为负无穷大（-inf）。
        dots1 = torch.where(mask1 > 0, dots, torch.full_like(dots, float('-inf'))) # 这个也要重新跑
        attn1 = dots1.softmax(dim=-1) #b*h*(n+1)*(n+1)
        # attn1 = dots1
        out1 = torch.einsum('bhij,bhjd->bhid', attn1, v) ##b*h*(n+1)*(c/h)
        out = rearrange(out1, 'b h n d -> b n (h d)') #b*(n+1)*c
        out = self.to_out(out) #b*(n+1)*c

###########################################################
        # attn = dots.softmax(dim=-1) #b*h*(n+1)*(n+1)

        # out = torch.einsum('bhij,bhjd->bhid', attn1, v) ##b*h*(n+1)*(c/h)
        # out = rearrange(out, 'b h n d -> b n (h d)') #b*(n+1)*c
        # out = self.to_out(out) #b*(n+1)*c
        return out

class sTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth): #depth为Transformer Encoder的堆叠次数，也即该部分深度，我们使用ModuleList既保持代码整洁又实现了模块堆叠
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))
    def forward(self, x, mask = None):

        for attn, ff in self.layers:
            x = attn(x, mask = mask) #[2, 251, 128]
            x = ff(x) #[2, 2001, 128]
        return x
