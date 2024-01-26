import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# from mm_modules.DCN.modules.deform_conv2d import DeformConv2dPack
import math

import pywt
from torch.autograd import Function
# from mmcv.cnn import build_norm_layer
import einops
from einops import rearrange
import numbers

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class DWMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., sr_ratio=1, alpha=0.5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        # print(type(w_ll), w_ll.shape, dim)
        # print(w_ll.dtype)
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        # self.filters = self.filters.to(dtype=torch.float16)
        self.filters = self.filters

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        # self.w_ll = self.w_ll.to(dtype=torch.float16)
        # self.w_lh = self.w_lh.to(dtype=torch.float16)
        # self.w_hl = self.w_hl.to(dtype=torch.float16)
        # self.w_hh = self.w_hh.to(dtype=torch.float16)

        self.w_ll = self.w_ll
        self.w_lh = self.w_lh
        self.w_hl = self.w_hl
        self.w_hh = self.w_hh

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

class WSDA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 window_size=2, alpha=0.5, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)

        self.dim = dim
        # self-attention heads in Lo-Fi
        self.w_heads = int(num_heads * alpha)
        # token dimension in Lo-Fi
        self.w_dim = self.w_heads * head_dim

        # self-attention heads in Hi-Fi
        self.s_heads = num_heads - self.w_heads
        # token dimension in Hi-Fi
        self.s_dim = self.s_heads * head_dim

        # local window size. The `s` in our paper.
        self.ws = window_size

        if self.ws == 1:
            # ws == 1 is equal to a standard multi-head self-attention
            self.s_heads = 0
            self.s_dim = 0
            self.w_heads = num_heads
            self.w_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        # Low frequence attention (Lo-Fi)
        if self.w_heads > 0:
            # if self.ws != 1:
            #     self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            # self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            # self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            # self.l_proj = nn.Linear(self.l_dim, self.l_dim)
            self.dwt = DWT_2D(wave='haar')
            self.idwt = IDWT_2D(wave='haar')

            self.reduce = nn.Sequential(
                nn.Conv2d(self.dim, self.w_dim // 4, kernel_size=1, padding=0, stride=1),
                # build_norm_layer(dict(type='BN', requires_grad=False), dim//4)[1],
                nn.BatchNorm2d(self.w_dim // 4, affine=True),
                nn.ReLU(inplace=True),
            )

            self.filter = nn.Sequential(
                nn.Conv2d(self.w_dim, self.w_dim, kernel_size=3, padding=1, stride=1, groups=1),
                # build_norm_layer(dict(type='BN', requires_grad=False), dim)[1],
                nn.BatchNorm2d(self.w_dim, affine=True),
                nn.ReLU(inplace=True),
            )

            self.w_kv_embed = nn.Conv2d(self.w_dim, self.w_dim,
                                        kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
            self.w_q = nn.Linear(self.dim, self.w_dim, bias=qkv_bias)
            self.w_kv = nn.Sequential(
                nn.LayerNorm(self.w_dim),
                nn.Linear(self.w_dim, self.w_dim * 2)
            )
            self.w_proj = nn.Linear(self.w_dim + self.w_dim // 4, self.w_dim)

        # High frequence attention (Hi-Fi)
        if self.s_heads > 0:
            self.s_qkv = nn.Linear(self.dim, self.s_dim * 3, bias=qkv_bias)
            self.s_proj = nn.Linear(self.s_dim, self.s_dim)


    def sa(self, x):
        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.s_qkv(x).reshape(B, total_groups, -1, 3, self.s_heads, self.s_dim // self.s_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.s_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.s_dim)
        x = self.s_proj(x)
        return x

    def wa(self, x):
        B, H, W, C = x.shape

        q = self.w_q(x).reshape(B, H * W, self.w_heads, self.w_dim // self.w_heads).permute(0, 2, 1, 3)
        x = x.permute(0, 3, 1, 2)
        x_dwt = self.dwt(self.reduce(x))
        x_dwt = self.filter(x_dwt)

        x_idwt = self.idwt(x_dwt)
        x_idwt = x_idwt.view(B, -1, x_idwt.size(-2) * x_idwt.size(-1)).transpose(1, 2)

        kv = self.w_kv_embed(x_dwt).reshape(B, self.w_dim, -1).permute(0, 2, 1)
        kv = self.w_kv(kv).reshape(B, -1, 2, self.w_heads, self.w_dim // self.w_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H*W, self.w_dim)
        x = self.w_proj(torch.cat([x, x_idwt], dim=-1))
        x = x.reshape(B, H, W, self.w_dim)
        return x

    def forward(self, x, H, W):
        B, N, C = x.shape

        x = x.reshape(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        if self.s_heads == 0:
            x = self.wa(x)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :]
            return x.reshape(B, N, C)

        if self.w_heads == 0:
            x = self.sa(x)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :]
            return x.reshape(B, N, C)

        sa_out = self.sa(x)
        wa_out = self.wa(x)
        # print('h', hifi_out.shape)
        # print('l', lofi_out.shape)

        if pad_r > 0 or pad_b > 0:
            x = torch.cat((sa_out[:, :H, :W, :], wa_out[:, :H, :W, :]), dim=-1)
        else:
            x = torch.cat((sa_out, wa_out), dim=-1)

        x = x.reshape(B, N, C)
        return x

    def flops(self, N):
        H = int(N ** 0.5)
        Hp = Wp = self.ws * math.ceil(H / self.ws)

        Np = Hp * Wp

        # For Hi-Fi
        # qkv
        hifi_flops = Np * self.dim * self.s_dim * 3
        nW = Np / self.ws / self.ws
        window_len = self.ws * self.ws
        # q @ k and attn @ v
        window_flops = window_len * window_len * self.s_dim * 2
        hifi_flops += nW * window_flops
        # projection
        hifi_flops += Np * self.s_dim * self.s_dim

        # for Lo-Fi
        # q
        lofi_flops = Np * self.dim * self.w_dim
        # H = int(Np ** 0.5)
        kv_len = (Hp // self.ws) ** 2
        # k, v
        lofi_flops += kv_len * self.dim * self.w_dim * 2
        # q @ k and attn @ v
        lofi_flops += Np * self.w_dim * kv_len * 2
        # projection
        lofi_flops += Np * self.w_dim * self.w_dim

        return hifi_flops + lofi_flops


class Block(nn.Module):
    """ Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, input_resolution, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, local_ws=1, alpha=0.5, sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)

        self.attn = WSDA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         attn_drop=attn_drop, proj_drop=drop, window_size=local_ws, alpha=alpha, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DWMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.mlp = FeedForward(dim, mlp_ratio)


    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

# Upsample Block
class TransposeConv(nn.Module):
    def __init__(self, dim):
        super(TransposeConv, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(dim, dim//2, kernel_size=2, stride=2),
        )
        self.in_channel = dim
        self.out_channel = dim//2

    def forward(self, x):
        x = self.deconv(x)
        return x

# Downsample Block
class Downsample(nn.Module):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = dim
        self.out_channel = dim*2

    def forward(self, x):
        x = self.conv(x) # B H*W C
        return x
# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

class Reduce(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduce = nn.Linear(in_channels, out_channels)

    def forward(self, x, skip_x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        skip_x = skip_x.flatten(2).transpose(1, 2).contiguous()
        out = self.reduce(torch.cat([x, skip_x], -1))
        out = out.transpose(1, 2).contiguous().view(B, C, H, W)
        return out

# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, act_layer=nn.LeakyReLU):
        super(InputProj, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True)
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x

class OutputProj(nn.Module):
    def __init__(self, in_channel=64, hidden_channel=48,out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super(OutputProj, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1),
        )
        self.active = nn.Sigmoid()
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self,x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        # x = self.active(x)
        return x

class WSDLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 input_resolution,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 local_ws=1,
                 alpha=0.5,
                 sr_ratio=1
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.input_resolution = input_resolution
        # build blocks
        block = Block
        self.blocks = nn.ModuleList([
            block(
                dim=dim,
                num_heads=num_heads,
                input_resolution=input_resolution,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                local_ws=local_ws,
                alpha=alpha,
                sr_ratio=sr_ratio
            )
            for i in range(depth)])


    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class Prior_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(Prior_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.x_kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.prior_qv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.dwconv1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.dwconv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)

    def forward(self, x, x_p, h, w):
        b, n, c = x.shape
        x = x.transpose(1, 2).view(b, c, h, w)
        x_p = x_p.transpose(1, 2).view(b, c, h, w)

        x_kv = self.dwconv1(self.x_kv(x))
        x_k, x_v = x_kv.chunk(2, dim=1)

        prior_qv = self.dwconv2(self.prior_qv(x_p))
        prior_q, prior_v = prior_qv.chunk(2, dim=1)

        x_k = rearrange(x_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        x_v = rearrange(x_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        prior_q = rearrange(prior_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        prior_v = rearrange(prior_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        prior_q = torch.nn.functional.normalize(prior_q, dim=-1)
        x_k = torch.nn.functional.normalize(x_k, dim=-1)

        attn = (prior_q @ x_k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out_x = (attn @ x_v)
        out_p = (attn @ prior_v)

        out_x = rearrange(out_x, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_p = rearrange(out_p, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(torch.cat([out_x, out_p], dim=1))
        out = out.flatten(2).transpose(1, 2).contiguous()
        return out

class Prior_Block(nn.Module):
    def __init__(self, dim, num_heads,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.norm_p = norm_layer(dim)
        self.attn = Prior_Attention(dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DWMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.mlp = FeedForward(dim, mlp_ratio)

    def forward(self, x, x_p, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm_p(x_p), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class WSDformer(nn.Module):
    def __init__(self,
                 vis=False,
                 img_size=128,
                 in_chans=3,
                 features=64,
                 input_resolution=[128, 128],
                 embed_dim=[64, 128, 256, 512, 256, 128, 64],
                 depths=[4, 4, 4, 4, 4, 4, 4],
                 num_heads=[2, 4, 8, 16, 8, 4, 2],
                 window_size=8,
                 mlp_ratio=[8, 8, 4, 4, 4, 8, 8],
                 sr_ratio=[4, 2, 1, 1, 1, 2, 4],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 alpha=0.5,
                 local_ws=[8, 8, 8, 8, 8, 8, 8],
                 prior_heads=4,
                 prior_depths=1,
                 block_size_hr=(16, 16), block_size_lr=(8, 8),
                 grid_size_hr=(16, 16), grid_size_lr=(8, 8),
                 high_res_stages=2,
                 use_bias=True
                 ):
        super().__init__()
        # new from v2
        self.local_ws = local_ws
        self.alpha = alpha
        self.num_heads = num_heads
        self.features = features

        # cross-gating setting
        self.block_size_hr = block_size_hr
        self.block_size_lr = block_size_lr
        self.grid_size_hr = grid_size_hr
        self.grid_size_lr = grid_size_lr
        self.high_res_stages = high_res_stages
        self.bias = use_bias
        self.drop = drop_rate

        # self.fp16_enabled = True
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed(
        #     patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0],
        #     norm_layer=norm_layer if self.patch_norm else None)
        self.inputproj = InputProj(in_channel=3, out_channel=embed_dim[0])

        n_feats = embed_dim[0]
        blocks = 3
        self.res_extra1 = res_ch(n_feats, blocks)

        self.prior_blocks = nn.ModuleList([
            Prior_Block(dim=embed_dim[0],
                        num_heads=prior_heads)
            for i in range(prior_depths)])

        self.downsample_0 = Downsample(embed_dim[0])
        self.downsample_1 = Downsample(embed_dim[1])
        self.downsample_2 = Downsample(embed_dim[2])

        self.upsample_3 = TransposeConv(embed_dim[3])
        self.upsample_4 = TransposeConv(embed_dim[4])
        self.upsample_5 = TransposeConv(embed_dim[5])

        self.reduce_chan_3to4 = nn.Conv2d(4*embed_dim[4], embed_dim[4],kernel_size=(1,1),stride=(1,1),bias=self.bias)
        self.reduce_chan_4to5 = nn.Conv2d(4*embed_dim[5], embed_dim[5],kernel_size=(1,1),stride=(1,1),bias=self.bias)
        self.reduce_chan_5to6 = nn.Conv2d(4*embed_dim[6], embed_dim[6],kernel_size=(1,1),stride=(1,1),bias=self.bias)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = WSDLayer(
                dim=embed_dim[i_layer],
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                input_resolution=(input_resolution[0] // (2 ** i_layer),
                                  input_resolution[1] // (2 ** i_layer)),
                window_size=window_size,
                mlp_ratio=mlp_ratio[i_layer],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                local_ws=self.local_ws[i_layer],
                alpha=alpha,
                sr_ratio=sr_ratio[i_layer])
            self.layers.append(layer)

        self.outputproj = OutputProj(in_channel=embed_dim[6],
                                     hidden_channel=32,
                                     out_channel=3)

        self._freeze_stages()

        # cross gating
        # depth=2
        self.UpSampleRatio_0 = UpSampleRatio_4(1 * self.features, ratio=2 ** (-2), use_bias=self.bias)  # 0->2
        self.UpSampleRatio_1 = UpSampleRatio_2(2 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 1->2
        self.UpSampleRatio_2 = UpSampleRatio(4 * self.features, ratio=1, use_bias=self.bias)  # 2->2
        self.cross_gating_block_2 = CrossGatingBlock(cin_y=8 * features,
                                                     x_features=3 * (2 ** 2) * self.features,
                                                     num_channels=4 * features,
                                                     block_size=self.block_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                     grid_size=self.grid_size_hr if 2 < self.high_res_stages else self.block_size_lr,
                                                     upsample_y=True, use_bias=self.bias,
                                                     dropout_rate=self.drop)
        # depth=1
        self.UpSampleRatio_3 = UpSampleRatio_2(1 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 0->1
        self.UpSampleRatio_4 = UpSampleRatio(2 * self.features, ratio=2 ** (0), use_bias=self.bias)  # 1->1
        self.UpSampleRatio_5 = UpSampleRatio_1_2(4 * self.features, ratio=2, use_bias=self.bias)  # 2->1
        self.cross_gating_block_1 = CrossGatingBlock(cin_y=4 * features, x_features=3 * 2 * self.features,
                                                     num_channels=2 * features,
                                                     block_size=self.block_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                     grid_size=self.grid_size_hr if 1 < self.high_res_stages else self.block_size_lr,
                                                     upsample_y=True, use_bias=self.bias,
                                                     dropout_rate=self.drop)
        # depth=0
        self.UpSampleRatio_6 = UpSampleRatio(1 * self.features, ratio=1, use_bias=self.bias)  # 0->0
        self.UpSampleRatio_7 = UpSampleRatio_1_2(2 * self.features, ratio=2, use_bias=self.bias)  # 1->0
        self.UpSampleRatio_8 = UpSampleRatio_1_4(4 * self.features, ratio=4, use_bias=self.bias)  # 2->0
        self.cross_gating_block_0 = CrossGatingBlock(cin_y=2 * features, x_features=3 * self.features,
                                                     num_channels=self.features,
                                                     block_size=self.block_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                     grid_size=self.grid_size_hr if 0 < self.high_res_stages else self.block_size_lr,
                                                     upsample_y=True, use_bias=self.bias,
                                                     dropout_rate=self.drop)

        self.UpSampleRatio_9 = UpSampleRatio(4 * self.features, ratio=2 ** (0), use_bias=self.bias)  # 2->2
        self.UpSampleRatio_10 = UpSampleRatio_2(2 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 1->2
        self.UpSampleRatio_11 = UpSampleRatio_4(1 * self.features, ratio=2 ** (-2), use_bias=self.bias)  # 0->2

        self.UpSampleRatio_12 = UpSampleRatio_1_2(4 * self.features, ratio=2 ** (1), use_bias=self.bias)  # 2->1
        self.UpSampleRatio_13 = UpSampleRatio(2 * self.features, ratio=2 ** (0), use_bias=self.bias)  # 1->1
        self.UpSampleRatio_14 = UpSampleRatio_2(1 * self.features, ratio=2 ** (-1), use_bias=self.bias)  # 0->1

        self.UpSampleRatio_15 = UpSampleRatio_1_4(4 * self.features, ratio=4, use_bias=self.bias)  # 2->0
        self.UpSampleRatio_16 = UpSampleRatio_1_2(2 * self.features, ratio=2, use_bias=self.bias)  # 1->0
        self.UpSampleRatio_17 = UpSampleRatio(1 * self.features, ratio=1, use_bias=self.bias)  # 0->0

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Forward function."""
        input_p = x
        res_x = get_residue(x)
        res_feats = self.res_extra1(torch.cat((res_x, res_x, res_x), dim=1))

        # input_proj
        x_init = self.inputproj(x)
        B, C, H, W = x_init.shape
        x_init = x_init.flatten(2).transpose(1, 2).contiguous()
        res_feats = res_feats.flatten(2).transpose(1, 2).contiguous()

        for i, blk in enumerate(self.prior_blocks):
            x_s = blk(x_init, res_feats, H, W)
        x0 = x_s.transpose(1, 2).view(B, C, H, W)

        # encoder
        encs = []
        xe0 = self.layers[0](x0)
        encs.append(xe0)

        xe0_1 = self.downsample_0(xe0)
        xe1= self.layers[1](xe0_1)
        encs.append(xe1)

        xe1_2= self.downsample_1(xe1)
        xe2 = self.layers[2](xe1_2)
        encs.append(xe2)

        # bottleneck
        xe2_3 = self.downsample_2(xe2)
        xb3 = self.layers[3](xe2_3)
        global_feature = xb3

        # cross-gating skip connection
        skip_features = []
        for i in reversed(range(3)):  # 2, 1, 0 (循环倒序)
            if i == 2:
                # get multi-scale skip signals from cross-gating block
                signal0 = self.UpSampleRatio_0(encs[0])
                signal1 = self.UpSampleRatio_1(encs[1])
                signal2 = self.UpSampleRatio_2(encs[2])
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                # Use cross-gating to cross modulate features
                skips, global_feature = self.cross_gating_block_2(signal, global_feature)
                skip_features.append(skips)
            elif i == 1:
                # get multi-scale skip signals from cross-gating block
                signal0 = self.UpSampleRatio_3(encs[0])
                signal1 = self.UpSampleRatio_4(encs[1])
                signal2 = self.UpSampleRatio_5(encs[2])
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                # Use cross-gating to cross modulate features
                skips, global_feature = self.cross_gating_block_1(signal, global_feature)
                skip_features.append(skips)
            elif i == 0:
                # get multi-scale skip signals from cross-gating block
                signal0 = self.UpSampleRatio_6(encs[0])
                signal1 = self.UpSampleRatio_7(encs[1])
                signal2 = self.UpSampleRatio_8(encs[2])
                signal = torch.cat([signal0, signal1, signal2], dim=1)
                # Use cross-gating to cross modulate features
                skips, global_feature = self.cross_gating_block_0(signal, global_feature)
                skip_features.append(skips)

        # decoder
        xd3_4 = self.upsample_3(xb3)
        signal_d3_2 = self.UpSampleRatio_9(skip_features[0])
        signal_d3_1 = self.UpSampleRatio_10(skip_features[1])
        signal_d3_0 = self.UpSampleRatio_11(skip_features[2])
        signal_d3 = torch.cat([signal_d3_2, signal_d3_1, signal_d3_0, xd3_4], dim=1)
        xd3_4 = self.reduce_chan_3to4(signal_d3)
        xd4 = self.layers[4](xd3_4)

        xd4_5 = self.upsample_4(xd4)
        signal_d4_2 = self.UpSampleRatio_12(skip_features[0])
        signal_d4_1 = self.UpSampleRatio_13(skip_features[1])
        signal_d4_0 = self.UpSampleRatio_14(skip_features[2])
        signal_d4 = torch.cat([signal_d4_2, signal_d4_1, signal_d4_0, xd4_5], dim=1)
        xd4_5 = self.reduce_chan_4to5(signal_d4)
        xd5 = self.layers[5](xd4_5)

        xd5_6 = self.upsample_5(xd5)
        signal_d5_2 = self.UpSampleRatio_15(skip_features[0])
        signal_d5_1 = self.UpSampleRatio_16(skip_features[1])
        signal_d5_0 = self.UpSampleRatio_17(skip_features[2])
        signal_d5 = torch.cat([signal_d5_2, signal_d5_1, signal_d5_0, xd5_6], dim=1)
        xd5_6 = self.reduce_chan_5to6(signal_d5)
        xd6 = self.layers[6](xd5_6)

        # output_proj
        out = self.outputproj(xd6)
        return out+input_p

def get_residue(tensor , r_dim = 1):
    """
    return residue_channle (RGB)
    """
    # res_channel = []
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel

class convd(nn.Module):
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(convd, self).__init__()
        self.relu = nn.ReLU()
        self.padding = nn.ReflectionPad2d(kernel_size//2)
        self.conv = nn.Conv2d(inputchannel, outchannel, kernel_size, stride)
        self.ins = nn.InstanceNorm2d(outchannel, affine=True)
    def forward(self, x):
        x = self.conv(self.padding(x))
        # x= self.ins(x)
        x = self.relu(x)
        return x


class RB(nn.Module):
    def __init__(self, n_feats, nm='in'):
        super(RB, self).__init__()
        module_body = []
        for i in range(2):
            module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
            module_body.append(nn.ReLU())
        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.ReLU()
        self.se = SELayer(n_feats, 1)

    def forward(self, x):
        res = self.module_body(x)
        res = self.se(res)
        res += x
        return res


class RIR(nn.Module):
    def __init__(self, n_feats, n_blocks, nm='in'):
        super(RIR, self).__init__()
        module_body = [
            RB(n_feats) for _ in range(n_blocks)
        ]
        module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.module_body(x)
        res += x
        return self.relu(res)


class res_ch(nn.Module):
    def __init__(self, n_feats, blocks=2):
        super(res_ch, self).__init__()
        self.conv_init1 = convd(3, n_feats // 2, 3, 1)
        self.conv_init2 = convd(n_feats // 2, n_feats, 3, 1)
        self.extra = RIR(n_feats, n_blocks=blocks)

    def forward(self, x):
        x = self.conv_init2(self.conv_init1(x))
        x = self.extra(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GetSpatialGatingWeights(nn.Module):  # n, h, w, c
    """Get gating weights for cross-gating MLP block."""

    def __init__(self, num_channels, grid_size, block_size, input_proj_factor=2, use_bias=True, dropout_rate=0):
        super().__init__()
        self.num_channels = num_channels
        self.grid_size = grid_size
        self.block_size = block_size
        self.gh = self.grid_size[0]
        self.gw = self.grid_size[1]
        self.fh = self.block_size[0]
        self.fw = self.block_size[1]
        self.input_proj_factor = input_proj_factor
        self.use_bias = use_bias
        self.drop = dropout_rate
        self.LayerNorm_in = Layer_norm_process(self.num_channels)
        self.in_project = nn.Linear(self.num_channels, self.num_channels * self.input_proj_factor, bias=self.use_bias)
        # self.gelu = nn.GELU(approximate='tanh')
        self.gelu = nn.GELU()
        self.Dense_0 = nn.Linear(self.gh * self.gw, self.gh * self.gw, bias=self.use_bias)
        self.Dense_1 = nn.Linear(self.fh * self.fw, self.fh * self.fw, bias=self.use_bias)
        self.out_project = nn.Linear(self.num_channels * self.input_proj_factor, self.num_channels, bias=self.use_bias)
        self.dropout = nn.Dropout(self.drop)

    def forward(self, x):
        _, h, w, _ = x.shape
        # input projection
        x = self.LayerNorm_in(x)
        x = self.in_project(x)  # channel projection
        x = self.gelu(x)
        c = x.size(-1) // 2
        u, v = torch.split(x, c, dim=-1)
        # get grid MLP weights
        fh, fw = h // self.gh, w // self.gw
        u = block_images_einops(u, patch_size=(fh, fw))  # n, (gh gw) (fh fw) c
        # print('222555', u.shape)
        u = u.permute(0, 3, 2, 1)  # n, c, (fh fw) (gh gw)
        # print('222', u.shape, fh, fw, self.gh, self.gw)
        u = self.Dense_0(u)
        u = u.permute(0, 3, 2, 1)  # n, (gh gw) (fh fw) c
        u = unblock_images_einops(u, grid_size=(self.gh, self.gw), patch_size=(fh, fw))
        # get block MLP weights
        gh, gw = h // self.fh, w // self.fw
        v = block_images_einops(v, patch_size=(self.fh, self.fw))  # n, (gh gw) (fh fw) c
        # print('333555', v.shape)
        v = v.permute(0, 1, 3, 2)  # n (gh gw) c (fh fw)
        # print('333', v.shape, gh, gw, self.fh, self.fw)
        v = self.Dense_1(v)
        v = v.permute(0, 1, 3, 2)  # n, (gh gw) (fh fw) c
        v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(self.fh, self.fw))

        x = torch.cat([u, v], dim=-1)
        x = self.out_project(x)
        x = self.dropout(x)
        return x

class CrossGatingBlock(nn.Module):  #input shape: n, c, h, w
    """Cross-gating MLP block."""
    def __init__(self, x_features, num_channels, block_size, grid_size, cin_y=0,upsample_y=True, use_bias=True, use_global_mlp=True, dropout_rate=0):
        super().__init__()
        self.cin_y = cin_y
        self.x_features = x_features
        self.num_channels = num_channels
        self.block_size = block_size
        self.grid_size = grid_size
        self.upsample_y = upsample_y
        self.use_bias = use_bias
        self.use_global_mlp = use_global_mlp
        self.drop = dropout_rate
        self.ConvTranspose_0 = nn.ConvTranspose2d(self.cin_y,self.num_channels,kernel_size=(2,2),stride=2,bias=self.use_bias)
        self.Conv_0 = nn.Conv2d(self.x_features, self.num_channels, kernel_size=(1,1),stride=1, bias=self.use_bias)
        self.Conv_1 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=(1,1),stride=1, bias=self.use_bias)
        self.LayerNorm_x = Layer_norm_process(self.num_channels)
        self.in_project_x = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        # self.gelu1 = nn.GELU(approximate='tanh')
        self.gelu1 = nn.GELU()
        self.SplitHeadMultiAxisGating_x = GetSpatialGatingWeights(num_channels=self.num_channels,block_size=self.block_size,grid_size=self.grid_size,
            dropout_rate=self.drop,use_bias=self.use_bias)
        self.LayerNorm_y = Layer_norm_process(self.num_channels)
        self.in_project_y = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        # self.gelu2 = nn.GELU(approximate='tanh')
        self.gelu2 = nn.GELU()
        self.SplitHeadMultiAxisGating_y = GetSpatialGatingWeights(num_channels=self.num_channels,block_size=self.block_size,grid_size=self.grid_size,
            dropout_rate=self.drop,use_bias=self.use_bias)
        self.out_project_y = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.dropout1 = nn.Dropout(self.drop)
        self.out_project_x = nn.Linear(self.num_channels, self.num_channels, bias=self.use_bias)
        self.dropout2 = nn.Dropout(self.drop)
    def forward(self, x,y):
        # Upscale Y signal, y is the gating signal.
        if self.upsample_y:
                y = self.ConvTranspose_0(y)
        x = self.Conv_0(x)
        y = self.Conv_1(y)
        assert y.shape == x.shape
        x = x.permute(0,2,3,1)  #n,h,w,c
        y = y.permute(0,2,3,1)  #n,h,w,c
        shortcut_x = x
        shortcut_y = y
        # Get gating weights from X
        x = self.LayerNorm_x(x)
        x = self.in_project_x(x)
        # print('111', x.shape)
        x = self.gelu1(x)
        gx = self.SplitHeadMultiAxisGating_x(x)
        # Get gating weights from Y
        y = self.LayerNorm_y(y)
        y = self.in_project_y(y)
        y = self.gelu2(y)
        gy = self.SplitHeadMultiAxisGating_y(y)
        # Apply cross gating
        y = y * gx  ## gating y using x
        y = self.out_project_y(y)
        y = self.dropout1(y)
        y = y + shortcut_y
        x = x * gy  # gating x using y
        x = self.out_project_x(x)
        x = self.dropout2(x)
        x = x + y + shortcut_x  # get all aggregated signals
        return x.permute(0,3,1,2), y.permute(0,3,1,2)  #n,c,h,w

class Layer_norm_process(nn.Module):  #n, h, w, c
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros(c), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones(c), requires_grad=True)
        self.eps = eps
    def forward(self, feature):
        var_mean = torch.var_mean(feature, dim=-1, unbiased=False)
        mean = var_mean[1]
        var = var_mean[0]
        # layer norm process
        feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + self.eps)
        gamma = self.gamma.expand_as(feature)
        beta = self.beta.expand_as(feature)
        feature = feature * gamma + beta
        return feature

def block_images_einops(x, patch_size):  #n, h, w, c
  """Image to patches."""
  batch, height, width, channels = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
  return x

def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x


class UpSampleRatio_4(nn.Module):  # input shape: n,c,h,w.    c-->4c
    """Upsample features given a ratio > 0."""

    def __init__(self, features, b=0, ratio=1., use_bias=True):
        super().__init__()
        self.features = features
        self.ratio = ratio
        self.bias = use_bias
        self.Conv_0 = nn.Conv2d(self.features, 4 * self.features, kernel_size=(1, 1), stride=1, bias=self.bias)

    def forward(self, x):
        n, c, h, w = x.shape
        # x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear', antialias=True)
        x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear')
        x = self.Conv_0(x)
        return x


class UpSampleRatio_2(nn.Module):  # input shape: n,c,h,w.    c-->2c
    """Upsample features given a ratio > 0."""

    def __init__(self, features, b=0, ratio=1., use_bias=True):
        super().__init__()
        self.features = features
        self.ratio = ratio
        self.bias = use_bias
        self.Conv_0 = nn.Conv2d(self.features, 2 * self.features, kernel_size=(1, 1), stride=1, bias=self.bias)

    def forward(self, x):
        n, c, h, w = x.shape
        # x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear', antialias=True)
        x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear')
        x = self.Conv_0(x)
        return x


class UpSampleRatio(nn.Module):  # input shape: n,c,h,w.    c-->c
    """Upsample features given a ratio > 0."""

    def __init__(self, features, b=0, ratio=1., use_bias=True):
        super().__init__()
        self.features = features
        self.ratio = ratio
        self.bias = use_bias
        self.Conv_0 = nn.Conv2d(self.features, self.features, kernel_size=(1, 1), stride=1, bias=self.bias)

    def forward(self, x):
        x = self.Conv_0(x)
        return x


class UpSampleRatio_1_2(nn.Module):  # input shape: n,c,h,w.    c-->c/2
    """Upsample features given a ratio > 0."""

    def __init__(self, features, b=0, ratio=1., use_bias=True):
        super().__init__()
        self.features = features
        self.ratio = ratio
        self.bias = use_bias
        self.Conv_0 = nn.Conv2d(self.features, self.features // 2, kernel_size=(1, 1), stride=1, bias=self.bias)

    def forward(self, x):
        n, c, h, w = x.shape
        # x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear', antialias=True)
        x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear')
        x = self.Conv_0(x)
        return x


class UpSampleRatio_1_4(nn.Module):  # input shape: n,c,h,w.    c-->c/4
    """Upsample features given a ratio > 0."""

    def __init__(self, features, b=0, ratio=1., use_bias=True):
        super().__init__()
        self.features = features
        self.ratio = ratio
        self.bias = use_bias
        self.Conv_0 = nn.Conv2d(self.features, self.features // 4, kernel_size=(1, 1), stride=1, bias=self.bias)

    def forward(self, x):
        n, c, h, w = x.shape
        # x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear', antialias=True)
        x = F.interpolate(x, size=(int(h * self.ratio), int(w * self.ratio)), mode='bilinear')
        x = self.Conv_0(x)
        return x