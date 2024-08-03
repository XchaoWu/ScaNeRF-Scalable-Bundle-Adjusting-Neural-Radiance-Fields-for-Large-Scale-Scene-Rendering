import numpy as np
import torch 
import torch.nn as nn 
import math 
from easydict import EasyDict as edict
import sys 

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def sh_encoding(deg, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.

    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]

    Returns:
        [..., C]
    """

    assert deg <= 4 and deg >= 0

    result = []

    result += [torch.ones_like(dirs[...,:1]) * C0]

    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result += [C1 * y, C1 * z, C1 * x]
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result += [C2[0] * xy, C2[1] * yz, C2[2] * (2.0 * zz - xx - yy), C2[3] * xz, C2[4] * (xx - yy)]

            if deg > 2:
                result += [C3[0] * y * (3 * xx - yy), C3[1] * xy * z, C3[2] * y * (4 * zz - xx - yy), 
                        C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
                        C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)]

                if deg > 3:
                    result += [C4[0] * xy * (xx - yy), C4[1] * yz * (3 * xx - yy), C4[2] * xy * (7 * zz - 1),
                                C4[3] * yz * (7 * zz - 3), C4[4] * (zz * (35 * zz - 30) + 3), C4[5] * xz * (7 * zz - 3),
                                C4[6] * (xx - yy) * (7 * zz - 1), C4[7] * xz * (xx - 3 * yy), C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))]
    return torch.cat(result, -1)

class Gaussian_Act(nn.Module):
    def __init__(self, sigma=0.1):
        super(Gaussian_Act, self).__init__()
        self.item = 1./ (-2 * (sigma ** 2))
    def forward(self, x):
        return torch.exp( (x ** 2) * self.item)

class GaussianAct(nn.Module):
    def __init__(self):
        super(GaussianAct, self).__init__()
    def forward(self, x, sigma):
        item = 1./ (-2 * (sigma ** 2))
        return torch.exp( (x ** 2) * item)

class Positional_Encoding(nn.Module):
    def __init__(self, L):
        super(Positional_Encoding, self).__init__()
        self.L = L 
    def embed(self, x, L):
        rets = [x]
        for i in range(L):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn(2.**i*x))
        return torch.cat(rets, -1)   
    def forward(self, x):
        return self.embed(x, self.L)

class Weighted_Positional_Encoding(nn.Module):
    def __init__(self, L):
        super(Weighted_Positional_Encoding, self).__init__()
        self.L = L 
    def embed(self, x, L):
        rets = [x]
        for i in range(L):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn(2.**i*x))
        return torch.cat(rets, -1)   
    def forward(self, inputs, **kwargs):
        embed_x = self.embed(inputs, self.L) # B x [3 + L x 2 x 3]
        alpha = (kwargs['global_step'] - kwargs['start']) / (kwargs['end'] - kwargs['start']) * self.L 
        alpha = max(min(alpha, self.L), 0)
        k = torch.arange(self.L,dtype=torch.float32,device=inputs.device)
        weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
        in_channel = embed_x.shape[1] // (1 + 2 * self.L)
        embed_x[:,in_channel:] *= weight[:,None].repeat(1, in_channel * 2).reshape(1,-1)
        return embed_x


class GeneralMLP(nn.Module):
    def __init__(self, num_in, num_out, activation,
                 hiden_depth=4, hiden_width=64, output_act=False):
        super(GeneralMLP, self).__init__()
        
        assert(hiden_depth >= 1)

        if hiden_depth == 1:
            layers = [nn.Linear(num_in, num_out)]
        else:
            layers = [nn.Linear(num_in, hiden_width)]
            layers.append(activation)
            for i in range(hiden_depth-2):
                layers.append(nn.Linear(hiden_width, hiden_width))
                layers.append(activation)
            layers.append(nn.Linear(hiden_width, num_out))
        if output_act:
            layers.append(activation)
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class ShallowMLP(nn.Module):
    """
    feature 24 + pos 3 + dir 3 
    """
    def __init__(self, in_channel):
        super(ShallowMLP, self).__init__()

        self.Spatial_MLP = GeneralMLP(in_channel, 64, Gaussian_Act(0.1), 2, 64)
        self.sigma_layer = GeneralMLP(32, 1, nn.Softplus(), 1, None, True)
        self.diffuse_layer = GeneralMLP(32, 3, nn.Sigmoid(), 1, None, True)
        self.tint_layer = GeneralMLP(32, 3, nn.Sigmoid(), 1, None, True)

        self.Directional_MLP = GeneralMLP(32+16, 3, Gaussian_Act(0.1), 3, 64)

        self.color_act = nn.Sigmoid()


    def inference_sigma(self, x):
        H = self.Spatial_MLP(x)
        return self.sigma_layer(H[..., :32])

    def forward(self, x, **kwargs):

        features = x[..., :-3]
        viewdirs = x[..., -3:] 

        viewdirs = viewdirs / (viewdirs.norm(2, dim=-1, keepdim=True) + 1e-8)

        H = self.Spatial_MLP(features * kwargs["weight_feature"])

        sigma = self.sigma_layer(H[..., :32])
        tint = self.tint_layer(H[..., :32])
        c_d = self.diffuse_layer(H[..., :32])

        sh_embeddings = sh_encoding(3, viewdirs)
        c_s = self.color_act(self.Directional_MLP(torch.cat([H[..., 32:], sh_embeddings], -1)))


        out = {"diffuse": c_d, "specular": c_s ,"sigma":sigma, "tint": tint}
        return out





def init_model(model, mode = 'default'):
    assert mode in ['xavier', 'kaiming', 'zeros', 'default', "small"]
    def kaiming_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            layer.bias.data.fill_(0.)
    def xavier_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0.)
    def zeros_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            layer.bias.data.fill_(0.)
    def small_init(layer):
        if isinstance(layer, nn.Linear):
            nn.init.ones_(layer.weight) * 0.0001
            layer.bias.data.fill_(0.0001)
    if mode == 'default':
        return model 
    elif mode == 'kaiming':
        model.apply(kaiming_init)
        print('\n====== Kaiming Init ======\n')
    elif mode == 'xavier':
        model.apply(xavier_init)
        print('\n====== Xavier Init ======\n')
    elif mode == 'zeros':
        model.apply(zeros_init)
        print('\n====== zeros Init ======\n')
    elif mode == "small":
        model.apply(small_init)
        print('\n====== small Init ======\n')
    return model 