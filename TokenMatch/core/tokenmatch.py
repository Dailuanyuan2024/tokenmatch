import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import batch_episym
from Transformer import Transformer
from sTransformer import sTransformer
from einops import rearrange, repeat

class sTransM(nn.Module):
    def __init__(self, in_channel, out_channel, p_size,  T_depth, heads, dim_head, mlp_dim):
        super(sTransM, self).__init__()
        self.p_size = p_size 
        self.patch_to_embedding = nn.Linear(in_channel, out_channel) 
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channel)) 
        self.stransformer = sTransformer(out_channel, T_depth, heads, dim_head, mlp_dim)

    def forward(self, x): #bcn1
        _,_,hh,ww = x.size()#bcn1
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p_size, p2=self.p_size) #bnc
        x = self.patch_to_embedding(x) #bnc
        b, n, _ = x.size() #bnc
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b) #b1c
        x = torch.cat((cls_tokens, x), dim=1) #b*(n+1)*c
        x = self.stransformer(x)  #b*(n+1)*c
        x = rearrange(x[:, 1:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=self.p_size, p2=self.p_size, h=hh, w=ww) #bcn1
        return x 
    
class gen_cluster_token(nn.Module):
    def __init__(self,in_channel, out_channel, cluster, p_size):
        super(gen_cluster_token, self).__init__()
        self.p_size = p_size 
        self.cluster_token = nn.Parameter(torch.randn(1, cluster, out_channel)) 
        self.patch_to_embedding = nn.Linear(in_channel, out_channel)
        
    def forward(self, x): #x=bcn1
        _,_,hh,ww = x.size() #hh=n;ww=1
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p_size, p2=self.p_size) #bnc 
        x = self.patch_to_embedding(x)  #bnc
        b, n, _ = x.size() #b=6,n=2000
        cluster_tokens = repeat(self.cluster_token, '() n d -> b n d', b=b) #bmc
        x = torch.cat((x, cluster_tokens), dim=1) #b(n+m)c
        x = x.transpose(1,2).unsqueeze(3) #bc(n+m)1
        
        return x 
        

class clusterBlock_down(nn.Module):
    def __init__(self,in_channel):
        super(clusterBlock_down, self).__init__()
        
        self.attq = nn.Sequential(
            nn.Conv2d(in_channel,in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.attk = nn.Sequential(
            nn.Conv2d(in_channel,in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.attv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x): #b*c*(n+m)*1
        b, c, nc, _ = x.size() #bc(n+m)1
        xn = x[:,:,:-64,:] #bcn1
        xc = x[:,:,nc-64:,:] #bcm1 
        q1 = self.attq(xc).squeeze(3) #bcm
        k1 = self.attk(xn).squeeze(3) #bcn
        v1 = self.attv(xn).squeeze(3)  #bcn
        
        scores = torch.bmm(q1.transpose(1,2), k1) #bmn
        att = torch.softmax(scores, dim=2) #bmn
        out = torch.bmm(att,v1.transpose(1,2)) #bmc
        out = out.transpose(1,2).unsqueeze(3) #bcm1
        out = self.conv(out) #bcm1
        out = xc + self.gamma * out #bcm1
        return out
        




# class clusterBlock_up(nn.Module):
#     def __init__(self, in_channel, cluster):
#         nn.Module.__init__(self)
#         self.conv = nn.Sequential(
#                 nn.InstanceNorm2d(in_channel, eps=1e-3),
#                 nn.BatchNorm2d(in_channel),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channel,cluster, kernel_size=1))
        
#     def forward(self, x_up, x_down): #这个部分可以直接改成attention的那种;x_up=bc(n+m)1;x_down=bcm1
#         x_up = x_up[:,:,:-64,:] #bcn1
#         embed = self.conv(x_up) #bmn1
#         S = torch.softmax(embed, dim=1).squeeze(3) #bmn
#         out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3) #bcn

#         return out


class clusterBlock_up(nn.Module):
    def __init__(self,in_channel):
        super(clusterBlock_up, self).__init__()
        
        self.attq = nn.Sequential(
            nn.Conv2d(in_channel,in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.attk = nn.Sequential(
            nn.Conv2d(in_channel,in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.attv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel,in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_up, x_down): #这个部分可以直接改成attention的那种;x_up=bc(n+m)1;x_down=bcm1
        # b, c, nc, _ = x_up.size() #bc(n+m)1
        xn = x_up[:,:,:-64,:] #bcn1 
        xc = x_down #bcm1 
        q1 = self.attq(xn).squeeze(3) #bcn
        k1 = self.attk(xc).squeeze(3) #bcm
        v1 = self.attv(xc).squeeze(3)  #bcm
        
        scores = torch.bmm(q1.transpose(1,2), k1) #bnm
        att = torch.softmax(scores, dim=2) #bnm
        out = torch.bmm(att,v1.transpose(1,2)) #bnc
        out = out.transpose(1,2).unsqueeze(3) #bcn1
        out = self.conv(out) #bcn1
        out = xn + self.gamma * out #bcn1
        return out

    
# class LSCC(nn.Module):
#     def __init__(self, inchannel, outchannel):
#         super(LSCC, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(outchannel, outchannel, 3, padding=1),
#             nn.InstanceNorm2d(outchannel),
#             nn.BatchNorm2d(outchannel),
#         )
    
        

#     def forward(self, x): #x=bcm1
#         out = self.left(x) #bcm1
#         out = out + x #bcm1
#         return torch.relu(out) #bcm1

class sGCN(nn.Module):
    def __init__(self, in_channel):
        super(sGCN, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )
        # self.linear_1 = nn.Conv2d(in_channel, 1, (1, 1))

    def gcn(self, x,w):
        B, _, N, _ = x.size() 

        with torch.no_grad():
            # w = self.linear_1(x) #b1m1
            w = torch.relu(torch.tanh(w)).unsqueeze(-1)  
            # w = torch.relu(torch.tanh(w)).squeeze(-1) #b1m
            A = torch.bmm(w, w.transpose(1, 2)) #bmm
            mask1 = torch.zeros(B, N, N,device=x.device, requires_grad=False) #b*h*m*m   
            mask2 = torch.zeros(B, N, N,device=x.device, requires_grad=False) #b*h*m*m
            mask3 = torch.zeros(B, N, N,device=x.device, requires_grad=False) #b*h*m*m
            mask4 = torch.zeros(B, N, N,device=x.device, requires_grad=False) #b*h*m*m
                 
            index = torch.topk(A, k=int((N*9)/10), dim=-1, largest=True)[1] #b*h*(n+1)*(n+1)/2;从指定维度 dim 上选择前 k 个最大的元素。
            mask1.scatter_(-1, index, 1.) #b*h*(n+1)*(n+1),里面有0和1；将 mask1 中的所有 index 所对应的位置设置为 1.0，其余位置保持为 0.0
            A1 = torch.where(mask1 > 0, A, torch.full_like(A, float(0))) #b
            
            index = torch.topk(A, k=int((N*8)/10), dim=-1, largest=True)[1] #b*h*(n+1)*(n+1)/2;从指定维度 dim 上选择前 k 个最大的元素。
            mask2.scatter_(-1, index, 1.) #b*h*(n+1)*(n+1),里面有0和1；将 mask1 中的所有 index 所对应的位置设置为 1.0，其余位置保持为 0.0
            A2 = torch.where(mask2 > 0, A, torch.full_like(A, float(0))) #b
            
            index = torch.topk(A, k=int((N*7)/10), dim=-1, largest=True)[1] #b*h*(n+1)*(n+1)/2;从指定维度 dim 上选择前 k 个最大的元素。
            mask3.scatter_(-1, index, 1.) #b*h*(n+1)*(n+1),里面有0和1；将 mask1 中的所有 index 所对应的位置设置为 1.0，其余位置保持为 0.0
            A3 = torch.where(mask3 > 0, A, torch.full_like(A, float(0))) #b
            
            index = torch.topk(A, k=int((N*6)/10), dim=-1, largest=True)[1] #b*h*(n+1)*(n+1)/2;从指定维度 dim 上选择前 k 个最大的元素。
            mask4.scatter_(-1, index, 1.) #b*h*(n+1)*(n+1),里面有0和1；将 mask1 中的所有 index 所对应的位置设置为 1.0，其余位置保持为 0.0
            A4 = torch.where(mask4 > 0, A, torch.full_like(A, float(0))) #b
            
            attn1 = A1.softmax(dim=-1) #b*h*(n+1)*(n+1)
            attn2 = A2.softmax(dim=-1) #b*h*(n+1)*(n+1)
            attn3 = A3.softmax(dim=-1) #b*h*(n+1)*(n+1)
            attn4 = A4.softmax(dim=-1)
            
            A= attn1 +attn2+attn3+attn4
            
            I = torch.eye(N).unsqueeze(0).to(x.device).detach() #1mm
            A = A + I #bmm
            D_out = torch.sum(A, dim=-1) #bm 
            D = (1 / D_out) ** 0.5 #bm
            D = torch.diag_embed(D)  #bmm
            L = torch.bmm(D, A) #bmm
            L = torch.bmm(L, D) #bmm
        out = x.squeeze(-1).transpose(1, 2).contiguous() #bmc
        out = torch.bmm(L, out).unsqueeze(-1) #bmc1
        out = out.transpose(1, 2).contiguous() #bcm1

        return out

    def forward(self, x,w): #bcm1
        out = self.gcn(x,w) #bcm1
        out = self.conv(out) #bcm1
        return out

    
class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResNet_Block, self).__init__()
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x): #bcn1
        x1 = self.right(x) #bcn1
        out = self.left(x) #bcn1
        out = out + x1 #bcn1
        return torch.relu(out)
    

class TransM(nn.Module):
    def __init__(self, in_channel, out_channel, p_size,  T_depth, heads, dim_head, mlp_dim):
        super(TransM, self).__init__()
        self.p_size = p_size #1
        self.patch_to_embedding = nn.Linear(in_channel, out_channel) 
        self.transformer = Transformer(out_channel, T_depth, heads, dim_head, mlp_dim)

    def forward(self, x): #bc(n+m)1
        _,_,hh,ww = x.size() #bc(n+m)1 
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.p_size, p2=self.p_size) #b(n+m)c 
        x = self.patch_to_embedding(x) #b(n+m)c
        b, n, _ = x.size()#b=6;n=n+m=2064
        x = self.transformer(x) #b(n+m)c  
        x = x.transpose(1,2).unsqueeze(3) #bc(n+m)1
        return x 


class Tokenmatch_Block(nn.Module):
    def __init__(self, net_channels, input_channel,cluster):
        nn.Module.__init__(self)
        channels = net_channels
        cluster = cluster
        super(Tokenmatch_Block, self).__init__()


        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, channels, (1, 1)), #4或6 → 128
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        
        self.l0 = []
        for _ in range(4):
            self.l0.append(TransM(in_channel=128, out_channel=128, p_size=1,  T_depth=2,
                             heads=4, dim_head=32, mlp_dim=128))       
        
        
        self.ge_cluster_token = nn.Sequential(gen_cluster_token(in_channel=128, out_channel=128, cluster=64, p_size=1))
        self.cluster_down = nn.Sequential(clusterBlock_down(channels))
        # self.filter = nn.Sequential(LSCC(channels, channels))
        self.filter = sGCN(channels)
        self.cluster_up = clusterBlock_up(channels)
        
        
        self.l1 = []
        for _ in range(1):
            self.l1.append(sTransM(in_channel=128, out_channel=128, p_size=1,  T_depth=2,
                             heads=4, dim_head=32, mlp_dim=128))        

        self.embed_1 = nn.Sequential(
            ResNet_Block(channels, channels),
        )
        self.linear_1 = nn.Conv2d(channels, 1, (1, 1))
        
        
        self.l0 = nn.Sequential(*self.l0)        
        self.l1 = nn.Sequential(*self.l1)
    

    def forward(self, data, xs): #data=b4n1;xs=b1n4
        batch_size, num_pts = data.shape[0], data.shape[2]
        
        out = self.conv(data) #bcn1 
        out_token = self.ge_cluster_token(out) #bc(n+m)1
 
        x0 = self.l0(out_token) #bc(n+m)1
        x_cd = self.cluster_down(x0) #bcm1
        w_cd = self.linear_1(x_cd).view(batch_size, -1) #bm
        x_f = self.filter(x_cd,w_cd) #bcm1  #这个部分要换一下
        x_cp = self.cluster_up(x0,x_f)  #bcn1
        x1 = self.l1(x_cp) #bcn1
        out1 = self.embed_1(x1)  #bcn1


        logits = torch.squeeze(torch.squeeze(self.linear_1(out1),3),1) #bn
        e_hat = weighted_8points(xs, logits)

        x1, x2 = xs[:,0,:,:2], xs[:,0,:,2:4]
        e_hat_norm = e_hat
        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)

        return logits, e_hat, residual


class Tokenmatch(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = config.iter_num
        self.side_channel = (config.use_ratio==2) + (config.use_mutual==2) 
        self.weights_init = Tokenmatch_Block(config.net_channels, 4+self.side_channel,config.cluster)
        self.weights_iter = [Tokenmatch_Block(config.net_channels, 6+self.side_channel,config.cluster) for _ in range(config.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)
        

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
     
        input = data['xs'].transpose(1,3)
        if self.side_channel > 0:
            sides = data['sides'].transpose(1,2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)

        res_logits, res_e_hat = [], []
        logits, e_hat, residual = self.weights_init(input, data['xs'])  #进入
        res_logits.append(logits), res_e_hat.append(e_hat)
        for i in range(self.iter_num):
            logits, e_hat, residual = self.weights_iter[i](
                torch.cat([input, residual.detach(), torch.relu(torch.tanh(logits)).reshape(residual.shape).detach()], dim=1),
                data['xs'])
            res_logits.append(logits), res_e_hat.append(e_hat)
        return res_logits, res_e_hat  

#############################################################

def batch_symeig(X):
    # Run symmetric eigendecomposition on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b, d, d)

    for batch_idx in range(X.shape[0]):
        e, v = torch.linalg.eigh(
            X[batch_idx, :, :],
            UPLO="U"
        )
        bv[batch_idx, :, :] = v

    bv = bv.cuda()
    return bv



def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)
    

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

        