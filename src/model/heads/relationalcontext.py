import torch
import torch.nn as nn

from . import utils_heads
from . import attention
from .base import BaseHead


class ATRCModule(nn.Module):

    def __init__(self,
                 tasks,
                 atrc_genotype,
                 in_channels,
                 inter_channels,
                 drop_rate=0.05,
                 zero_init=True):
        super().__init__()
        self.tasks = tasks
        self.atrc_genotype = atrc_genotype

        self.cp_blocks = nn.ModuleDict()
        for target in self.tasks:
            self.cp_blocks[target] = nn.ModuleDict()
            for source in self.tasks:
                if atrc_genotype[target][source] == 0:  # none
                    pass
                elif atrc_genotype[target][source] == 1:  # global
                    self.cp_blocks[target][source] = attention.GlobalContextAttentionBlock(in_channels,
                                                                                           inter_channels)
                elif atrc_genotype[target][source] == 2:  # local
                    self.cp_blocks[target][source] = attention.LocalContextAttentionBlock(in_channels,
                                                                                          inter_channels,
                                                                                          kernel_size=9)
                elif atrc_genotype[target][source] == 3:  # t-label
                    self.cp_blocks[target][source] = attention.LabelContextAttentionBlock(in_channels,
                                                                                          inter_channels,
                                                                                          context_type='tlabel')
                elif atrc_genotype[target][source] == 4:  # s-label
                    self.cp_blocks[target][source] = attention.LabelContextAttentionBlock(in_channels,
                                                                                          inter_channels,
                                                                                          context_type='slabel')
                else:
                    raise ValueError

        self.out_proj = nn.ModuleDict()
        self.bottleneck = nn.ModuleDict()
        for target in self.tasks:
            nr_sources = len(list(self.cp_blocks[target].keys()))
            if nr_sources > 0:
                self.out_proj[target] = utils_heads.ConvBNReLU(inter_channels * nr_sources,
                                                               in_channels,
                                                               kernel_size=1,
                                                               norm_layer=nn.BatchNorm2d,
                                                               activation_layer=None)
                self.bottleneck[target] = nn.Sequential(utils_heads.ConvBNReLU(in_channels * 2,
                                                                               in_channels,
                                                                               kernel_size=1,
                                                                               norm_layer=nn.BatchNorm2d,
                                                                               activation_layer=nn.ReLU),
                                                        nn.Dropout2d(drop_rate))
            else:
                self.bottleneck[target] = nn.Sequential(utils_heads.ConvBNReLU(in_channels,
                                                                               in_channels,
                                                                               kernel_size=1,
                                                                               norm_layer=nn.BatchNorm2d,
                                                                               activation_layer=nn.ReLU),
                                                        nn.Dropout2d(drop_rate))
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

    def forward(self, task_specific_feats, aux_pred, image):
        aux_prob = utils_heads.spatial_normalize_pred(aux_pred, image)
        atrc_out_feats = {}
        for t in self.tasks:
            cp_out = []
            for s in self.tasks:
                if self.atrc_genotype[t][s] == 0:
                    continue
                cp_out.append(self.cp_blocks[t][s](target_task_feats=task_specific_feats[t],
                                                   source_task_feats=task_specific_feats[s],
                                                   target_aux_prob=aux_prob[t],
                                                   source_aux_prob=aux_prob[s]))
            if len(cp_out) > 0:
                distilled = torch.cat([task_specific_feats[t], self.out_proj[t](torch.cat(cp_out, dim=1))], dim=1)
            else:
                distilled = task_specific_feats[t]
            atrc_out_feats[t] = self.bottleneck[t](distilled)
        return atrc_out_feats


class RelationalContextHead(BaseHead):

    def __init__(self, atrc_genotype, **kwargs):
        super().__init__(**kwargs)
        self.atrc_genotype = atrc_genotype
        self.head_endpoints = ['final', 'aux']
        out_channels = self.in_channels // 4
        att_channels = out_channels // 2

        self.bottleneck = nn.ModuleDict({t: utils_heads.ConvBNReLU(self.in_channels,
                                                                   out_channels,
                                                                   kernel_size=3,
                                                                   norm_layer=nn.BatchNorm2d,
                                                                   activation_layer=nn.ReLU)
                                         for t in self.tasks})
        self.aux_tasks = self.get_aux_tasks()
        self.aux_conv = nn.ModuleDict({t: utils_heads.ConvBNReLU(self.in_channels,
                                                                 self.in_channels,
                                                                 kernel_size=1,
                                                                 norm_layer=nn.BatchNorm2d,
                                                                 activation_layer=nn.ReLU)
                                       for t in self.aux_tasks})
        self.aux_logits = nn.ModuleDict({t: nn.Conv2d(self.in_channels,
                                                      self.task_channel_mapping[t]['aux'],
                                                      kernel_size=1,
                                                      bias=True)
                                         for t in self.aux_tasks})
        self.atrc_module = ATRCModule(self.tasks,
                                      self.atrc_genotype,
                                      out_channels,
                                      att_channels)
        self.final_logits = nn.ModuleDict({t: nn.Conv2d(out_channels,
                                                        self.task_channel_mapping[t]['final'],
                                                        kernel_size=1,
                                                        bias=True)
                                           for t in self.tasks})
        self.init_weights()

    def forward(self, inp, inp_shape, image, **kwargs):
        inp = self._transform_inputs(inp)
        task_specific_feats = {t: self.bottleneck[t](inp) for t in self.tasks}

        aux_inp = inp.detach()  # no backprop from aux heads
        aux_pred = {t: self.aux_logits[t](self.aux_conv[t](aux_inp)) for t in self.aux_tasks}

        atrc_out_feats = self.atrc_module(task_specific_feats, aux_pred, image)
        final_pred = {t: self.final_logits[t](atrc_out_feats[t]) for t in self.tasks}

        final_pred = {t: nn.functional.interpolate(
            final_pred[t], size=inp_shape, mode='bilinear', align_corners=False) for t in self.tasks}
        aux_pred = {t: nn.functional.interpolate(
            aux_pred[t], size=inp_shape, mode='bilinear', align_corners=False) for t in self.aux_tasks}
        return {'final': final_pred, 'aux': aux_pred}

    def get_aux_tasks(self):
        # to make sure we only compute aux maps when necessary
        aux_tasks = []
        for task in self.tasks:
            if any(self.atrc_genotype[task][source] == 3 for source in self.tasks):
                aux_tasks.append(task)
                continue
            if any(self.atrc_genotype[target][task] == 4 for target in self.tasks):
                aux_tasks.append(task)
                continue
        return aux_tasks
