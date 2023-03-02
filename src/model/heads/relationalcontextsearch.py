import torch
import torch.nn as nn

from . import utils_heads
from . import attention
from .base import BaseHead


class ConvexCombination(nn.Module):

    def __init__(self, op_list):
        super().__init__()
        self.op_list = op_list
        # also initialize weight for `none` op
        self.arch_param = nn.Parameter(torch.zeros(len(self.op_list) + 1))
        self.register_buffer('fixed_weights', None)

    def forward(self, gumbel_temp, mode='gumbel', **kwargs):
        if self.fixed_weights is None:
            if mode == 'gumbel':
                sort_arch_param = torch.topk(nn.functional.softmax(self.arch_param, dim=-1), 2)
                if sort_arch_param[0][0] - sort_arch_param[0][1] >= 0.3:
                    # end stochastic process irreversibly
                    self.fixed_weights = torch.zeros_like(
                        self.arch_param, requires_grad=False).scatter_(-1, sort_arch_param[1][0], 1.0)
                    weights = self.fixed_weights
                else:
                    weights = nn.functional.gumbel_softmax(self.arch_param, tau=gumbel_temp, hard=False, dim=-1)
            elif mode == 'argmax':
                index = self.arch_param.max(-1, keepdim=True)[1]
                weights = torch.zeros_like(self.arch_param, requires_grad=False).scatter_(-1, index, 1.0)
        else:
            weights = self.fixed_weights

        # weights[0] receives appropriate gradient through gumbel softmax
        out = sum(weights[i + 1] * op(**kwargs)for i, op in enumerate(self.op_list))
        return out


class ATRCSearchModule(nn.Module):

    def __init__(self,
                 tasks,
                 in_channels,
                 inter_channels,
                 zero_init=True):
        super().__init__()
        self.tasks = tasks

        self.cp_blocks = nn.ModuleDict()
        for target in self.tasks:
            self.cp_blocks[target] = nn.ModuleDict()
            for source in self.tasks:
                op_list = [
                    attention.GlobalContextAttentionBlock(in_channels,
                                                          inter_channels,
                                                          last_affine=False),
                    attention.LocalContextAttentionBlock(in_channels,
                                                         inter_channels,
                                                         kernel_size=9,
                                                         last_affine=False),
                    attention.LabelContextAttentionBlock(in_channels,
                                                         inter_channels,
                                                         context_type='tlabel',
                                                         last_affine=False)
                ]
                if target != source:
                    op_list.append(attention.LabelContextAttentionBlock(in_channels,
                                                                        inter_channels,
                                                                        context_type='slabel',
                                                                        last_affine=False))
                self.cp_blocks[target][source] = ConvexCombination(nn.ModuleList(op_list))

        self.out_proj = nn.ModuleDict({t: utils_heads.ConvBNReLU(inter_channels * len(self.tasks),
                                                                 in_channels,
                                                                 kernel_size=1,
                                                                 norm_layer=nn.BatchNorm2d,
                                                                 activation_layer=None)
                                       for t in self.tasks})
        self.bottleneck = nn.ModuleDict({t: utils_heads.ConvBNReLU(in_channels * 2,
                                                                   in_channels,
                                                                   kernel_size=1,
                                                                   norm_layer=nn.BatchNorm2d,
                                                                   activation_layer=nn.ReLU)
                                         for t in self.tasks})
        if zero_init:
            for m in self.out_proj.values():
                if m.use_norm:
                    # initialize weight of last norm
                    nn.init.constant_(m.bn.weight, 0)
                    nn.init.constant_(m.bn.bias, 0)
                else:
                    # initialize weight of last conv
                    nn.init.constant_(m.conv.weight, 0)
                    nn.init.constant_(m.conv.bias, 0)

    def forward(self, task_specific_feats, aux_pred, image, gumbel_temp, mode='gumbel'):
        aux_prob = utils_heads.spatial_normalize_pred(aux_pred, image)
        atrc_out_feats = {}
        for t in self.tasks:
            cp_out = []
            for s in self.tasks:
                cp_out.append(self.cp_blocks[t][s](target_task_feats=task_specific_feats[t],
                                                   source_task_feats=task_specific_feats[s],
                                                   target_aux_prob=aux_prob[t],
                                                   source_aux_prob=aux_prob[s],
                                                   gumbel_temp=gumbel_temp,
                                                   mode=mode))
            distilled = torch.cat([task_specific_feats[t], self.out_proj[t](torch.cat(cp_out, dim=1))], dim=1)
            atrc_out_feats[t] = self.bottleneck[t](distilled)
        return atrc_out_feats


class RelationalContextSearchHead(BaseHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head_endpoints = ['final', 'aux']
        out_channels = self.in_channels // 4
        att_channels = out_channels // 2

        self.bottleneck = nn.ModuleDict({t: utils_heads.ConvBNReLU(self.in_channels,
                                                                   out_channels,
                                                                   kernel_size=3,
                                                                   norm_layer=nn.BatchNorm2d,
                                                                   activation_layer=nn.ReLU)
                                         for t in self.tasks})
        self.aux_conv = nn.ModuleDict({t: utils_heads.ConvBNReLU(self.in_channels,
                                                                 self.in_channels,
                                                                 kernel_size=1,
                                                                 norm_layer=nn.BatchNorm2d,
                                                                 activation_layer=nn.ReLU)
                                       for t in self.tasks})
        self.aux_logits = nn.ModuleDict({t: nn.Conv2d(self.in_channels,
                                                      self.task_channel_mapping[t]['aux'],
                                                      kernel_size=1,
                                                      bias=True)
                                        for t in self.tasks})
        self.atrc_module = ATRCSearchModule(self.tasks,
                                            out_channels,
                                            att_channels)
        self.final_logits = nn.ModuleDict({t: nn.Conv2d(out_channels,
                                                        self.task_channel_mapping[t]['final'],
                                                        kernel_size=1,
                                                        bias=True)
                                           for t in self.tasks})
        self.init_weights()

    def forward(self, inp, inp_shape, image, gumbel_temp, mode='gumbel'):
        inp = self._transform_inputs(inp)
        task_specific_feats = {t: self.bottleneck[t](inp) for t in self.tasks}

        aux_inp = inp.detach()  # no backprop from aux heads
        aux_pred = {t: self.aux_logits[t](self.aux_conv[t](aux_inp)) for t in self.tasks}

        atrc_out_feats = self.atrc_module(task_specific_feats, aux_pred, image, gumbel_temp=gumbel_temp, mode=mode)
        final_pred = {t: self.final_logits[t](atrc_out_feats[t]) for t in self.tasks}

        final_pred = {t: nn.functional.interpolate(
            final_pred[t], size=inp_shape, mode='bilinear', align_corners=False) for t in self.tasks}
        aux_pred = {t: nn.functional.interpolate(
            aux_pred[t], size=inp_shape, mode='bilinear', align_corners=False) for t in self.tasks}
        return {'final': final_pred, 'aux': aux_pred}
