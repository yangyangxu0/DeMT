import torch
import torch.nn as nn

from . import utils_heads
from .base import BaseHead

from timm.models.layers import DropPath
from torch import Tensor
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv

from einops import rearrange
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair
import math
from einops.layers.torch import Rearrange


class DemtHead(BaseHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head_endpoints = ['final']
        out_channels = self.in_channels // 8
        dim_ = 256
        self.bottleneck = nn.ModuleDict({t: utils_heads.ConvBNReLU(dim_,
                                                                   out_channels,
                                                                   kernel_size=3,
                                                                   norm_layer=nn.BatchNorm2d,
                                                                   activation_layer=nn.ReLU)
                                         for t in self.tasks})
        self.final_logits = nn.ModuleDict({t: nn.Conv2d(out_channels,
                                                        self.task_channel_mapping[t]['final'],
                                                        kernel_size=1,
                                                        bias=True)
                                           for t in self.tasks})
        self.init_weights()

        self.defor_mixers = nn.ModuleList([DefMixer(dim_in=dim_, dim=dim_, depth=1)  for t in range (len(self.tasks))])

        self.linear1 = nn.Sequential(nn.Linear(self.in_channels, dim_), nn.LayerNorm(dim_))

        self.task_fusion = nn.MultiheadAttention(embed_dim=dim_, num_heads=4, dropout=0.)
        self.smlp = nn.Sequential(nn.Linear(dim_, dim_), nn.LayerNorm(dim_))
        self.smlp2 = nn.ModuleList([nn.Sequential(nn.Linear(dim_, dim_), nn.LayerNorm(dim_))  for t in range (len(self.tasks))])

        self.task_querys = nn.ModuleList([nn.MultiheadAttention(embed_dim=dim_, num_heads=4, dropout=0.)  for t in range (len(self.tasks))])


    def forward(self, inp, inp_shape, **kwargs):
        inp = self._transform_inputs(inp)   #bchw
        b, c, h, w = inp.shape
        inp = self.linear1(inp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        outs=[]
        for ind, defor_mixer in enumerate(self.defor_mixers):
            out = defor_mixer(inp)
            out = rearrange(out, "b c h w -> b (h w) c").contiguous()
            outs.append(out)

        task_cat = torch.cat(outs, dim=1)

        task_cat = self.task_fusion(task_cat, task_cat, task_cat)[0]
        task_cat = self.smlp(task_cat)

        outs_ls = []
        for ind, task_query in enumerate(self.task_querys):
            inp = outs[ind] + self.smlp2[ind](task_query(outs[ind] ,task_cat, task_cat)[0])
            outs_ls.append(rearrange(inp, "b (h w) c -> b c h w", h=h, w=w).contiguous())

        inp_dict = {t: outs_ls[idx] for idx, t in enumerate(self.tasks)}
        task_specific_feats = {t: self.bottleneck[t](inp_dict[t]) for t in self.tasks}
        final_pred = {t: self.final_logits[t](task_specific_feats[t]) for t in self.tasks}
        final_pred = {t: nn.functional.interpolate(
                        final_pred[t], size=inp_shape, mode='bilinear', align_corners=False) for t in self.tasks}
        return {'final': final_pred}


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x1=self.fn(x)
        return x1+x


class DefMixer(nn.Module):
    def __init__(self,dim_in, dim, depth=1, kernel_size=1):
        super(DefMixer, self).__init__()

        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        ChlSpl(dim, dim, (1, 3), 1, 0),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=kernel_size),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)],
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ChlSpl(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(ChlSpl, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))

        self.get_offset = Offset(dim=in_channels, kernel_size=3)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
    def gen_offset(self):
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
            input: Tensor[b,c,h,w]
        """
        offset_2 = self.get_offset(input)
        B, C, H, W = input.size()

        return deform_conv2d_tv(input, offset_2, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class Offset(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.p_conv = nn.Conv2d(dim, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=1)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        self.opt = nn.Conv2d(2*self.kernel_size*self.kernel_size, dim*2, kernel_size=3, padding=1, stride=1, groups=2)


    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def forward(self, x):
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2
        p = self._get_p(offset, dtype)  #1,18,107,140
        p =self.opt(p)
        return p


