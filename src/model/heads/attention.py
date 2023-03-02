import math
import torch
import torch.nn as nn

from . import utils_heads


class GlobalContextAttentionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, last_affine=True):
        super().__init__()
        self.eps = 1e-12
        self.query_project = nn.Sequential(utils_heads.ConvBNReLU(in_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU),
                                           utils_heads.ConvBNReLU(out_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=None),
                                           nn.Softplus())
        self.key_project = nn.Sequential(utils_heads.ConvBNReLU(in_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU),
                                         utils_heads.ConvBNReLU(out_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=None),
                                         nn.Softplus())
        self.value_project = utils_heads.ConvBNReLU(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    norm_layer=nn.BatchNorm2d,
                                                    activation_layer=nn.ReLU,
                                                    affine=last_affine)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_task_feats, source_task_feats, **kwargs):
        batch_size = target_task_feats.size(0)

        key = self.key_project(source_task_feats)  # b x d x h x w
        value = self.value_project(source_task_feats)  # b x m x h x w
        key = key.view(*key.shape[:2], -1)  # b x d x hw
        value = value.view(*value.shape[:2], -1)  # b x m x hw

        query = self.query_project(target_task_feats)  # b x d x h x w
        query = query.view(*query.shape[:2], -1)  # b x d x hw

        S = torch.matmul(value, key.permute(0, 2, 1))  # b x m x d
        Z = torch.sum(key, dim=2)  # b x d
        denom = torch.matmul(Z.unsqueeze(1), query)  # b x 1 x hw
        V = torch.matmul(S, query) / denom.clamp_min(self.eps)  # b x m x hw
        V = V.view(batch_size, -1, *target_task_feats.shape[2:]).contiguous()
        return V


class LocalContextAttentionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, last_affine=True):
        super().__init__()

        from .attention_ops import similarFunction, weightingFunction
        self.f_similar = similarFunction.apply
        self.f_weighting = weightingFunction.apply

        self.kernel_size = kernel_size
        self.query_project = nn.Sequential(utils_heads.ConvBNReLU(in_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU),
                                           utils_heads.ConvBNReLU(out_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU))
        self.key_project = nn.Sequential(utils_heads.ConvBNReLU(in_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU),
                                         utils_heads.ConvBNReLU(out_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU))
        self.value_project = utils_heads.ConvBNReLU(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    norm_layer=nn.BatchNorm2d,
                                                    activation_layer=nn.ReLU,
                                                    affine=last_affine)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_task_feats, source_task_feats, **kwargs):
        query = self.query_project(target_task_feats)
        key = self.key_project(source_task_feats)
        value = self.value_project(source_task_feats)

        weight = self.f_similar(query, key, self.kernel_size, self.kernel_size)
        weight = nn.functional.softmax(weight / math.sqrt(key.size(1)), -1)
        out = self.f_weighting(value, weight, self.kernel_size, self.kernel_size)
        return out


class LabelContextAttentionBlock(nn.Module):

    def __init__(self, in_channels, out_channels, context_type, last_affine=True):
        super().__init__()
        self.context_type = context_type
        self.query_project = nn.Sequential(utils_heads.ConvBNReLU(in_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU),
                                           utils_heads.ConvBNReLU(out_channels,
                                                                  out_channels,
                                                                  kernel_size=1,
                                                                  norm_layer=nn.BatchNorm2d,
                                                                  activation_layer=nn.ReLU))
        self.key_project = nn.Sequential(utils_heads.ConvBNReLU(in_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU),
                                         utils_heads.ConvBNReLU(out_channels,
                                                                out_channels,
                                                                kernel_size=1,
                                                                norm_layer=nn.BatchNorm2d,
                                                                activation_layer=nn.ReLU))
        self.value_project = utils_heads.ConvBNReLU(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    norm_layer=nn.BatchNorm2d,
                                                    activation_layer=nn.ReLU,
                                                    affine=last_affine)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, target_task_feats, source_task_feats, target_aux_prob, source_aux_prob):
        context = self.gather_context(source_task_feats, target_aux_prob, source_aux_prob)
        batch_size = target_task_feats.size(0)

        key = self.key_project(context)
        value = self.value_project(context)
        key = key.view(*key.shape[:2], -1)
        value = value.view(*value.shape[:2], -1).permute(0, 2, 1)

        query = self.query_project(target_task_feats)
        query = query.view(*query.shape[:2], -1).permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map /= query.shape[-1]**0.5
        sim_map = sim_map.softmax(dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *target_task_feats.shape[2:])
        return context

    def gather_context(self, source_feats, target_aux_prob, source_aux_prob):
        if self.context_type == 'tlabel':
            batch_size, channels = source_feats.shape[:2]
            source_feats = source_feats.view(batch_size, channels, -1)
            source_feats = source_feats.permute(0, 2, 1)
            cxt = torch.matmul(target_aux_prob, source_feats)
            context = cxt.permute(0, 2, 1).contiguous().unsqueeze(3) 
        elif self.context_type == 'slabel':
            batch_size, channels = source_feats.shape[:2]
            source_feats = source_feats.view(batch_size, channels, -1)
            source_feats = source_feats.permute(0, 2, 1)
            cxt = torch.matmul(source_aux_prob, source_feats)
            context = cxt.permute(0, 2, 1).contiguous().unsqueeze(3)
        return context
