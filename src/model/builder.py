import os
import json
from functools import partial
import torch
import torch.nn as nn

from . import backbones
from . import criterion


def get_backbone(model_backbone,
                 pretrained=True):
    if model_backbone == 'HRNet18-S':
        backbone = backbones.hrnetv2p_w18_s(pretrained=pretrained)
        idx_to_planes = {0: 18, 1: 36, 2: 72, 3: 144}
    elif model_backbone == 'HRNet48':
        backbone = backbones.hrnetv2p_w48(pretrained=pretrained)
        idx_to_planes = {0: 48, 1: 96, 2: 192, 3: 384}
    elif model_backbone == 'Swin-B':
        backbone = backbones.swin_b(pretrained=pretrained)
        idx_to_planes = {0: 128, 1: 256, 2: 512, 3: 1024}
    elif model_backbone == 'Swin-S':
        backbone = backbones.swin_s(pretrained=pretrained)
        idx_to_planes = {0: 96, 1: 192, 2: 384, 3: 768}
    elif model_backbone == 'Swin-T':
        backbone = backbones.swin_t(pretrained=pretrained)
        idx_to_planes = {0: 96, 1: 192, 2: 384, 3: 768}
    else:
        raise ValueError

    backbone.idx_to_planes = idx_to_planes

    return backbone


def get_head(head_name,
             in_index,
             idx_to_planes,
             tasks,
             task_channel_mapping,
             atrc_genotype_path=None):
    in_index = [int(i) for i in in_index.split(',')]
    in_index = in_index[0] if len(in_index) == 1 else in_index

    if head_name == 'DemtHead':
        from .heads.demt_head import DemtHead
        partial_head = partial(DemtHead)

    elif 'RelationalContextHead' in head_name:
        from .heads.relationalcontext import RelationalContextHead
        if head_name == 'GlobalRelationalContextHead':
            atrc_genotype = {t: {a: 1 for a in tasks} for t in tasks}
        elif head_name == 'LocalRelationalContextHead':
            atrc_genotype = {t: {a: 2 for a in tasks} for t in tasks}
        elif head_name == 'TLabelRelationalContextHead':
            atrc_genotype = {t: {a: 3 for a in tasks} for t in tasks}
        elif head_name == 'SLabelRelationalContextHead':
            atrc_genotype = {t: {a: 4 for a in tasks} for t in tasks}
        elif head_name == 'AdaptiveTaskRelationalContextHead':
            assert os.path.isfile(atrc_genotype_path), \
                'When using ATRC, a path to a valid genotype json file needs to be supplied ' \
                'via `--model.atrc_genotype_path path/to/genotype.json`'
            with open(atrc_genotype_path) as f:
                atrc_genotype = json.load(f)['data']
        else:
            raise ValueError
        partial_head = partial(RelationalContextHead, atrc_genotype=atrc_genotype)
    elif head_name == 'AdaptiveTaskRelationalContextSearchHead':
        from .heads.relationalcontextsearch import RelationalContextSearchHead
        partial_head = partial(RelationalContextSearchHead)
    head = partial_head(tasks=tasks,
                        in_index=in_index,
                        idx_to_planes=idx_to_planes,
                        task_channel_mapping=task_channel_mapping)
    return head


def get_criterion(tasks, head_endpoints, edge_pos_weight=None, normals_centroids=None):
    loss_dict = nn.ModuleDict()
    loss_task_weights = {}
    for task in tasks:
        if task == 'semseg':
            loss_dict[task] = nn.ModuleDict()
            loss_task_weights[task] = 1.0
            for p in head_endpoints:
                if p == 'aux':
                    loss_dict[task][p] = criterion.CrossEntropyLoss()
                else:
                    loss_dict[task][p] = criterion.CrossEntropyLoss()
        elif task == 'human_parts':
            loss_dict[task] = nn.ModuleDict()
            loss_task_weights[task] = 2.0
            for p in head_endpoints:
                if p == 'aux':
                    loss_dict[task][p] = criterion.CrossEntropyLoss()
                else:
                    loss_dict[task][p] = criterion.CrossEntropyLoss()
        elif task == 'sal':
            loss_dict[task] = nn.ModuleDict()
            loss_task_weights[task] = 5.0
            for p in head_endpoints:
                if p == 'aux':
                    loss_dict[task][p] = criterion.CrossEntropyLoss(balanced=True)
                else:
                    loss_dict[task][p] = criterion.CrossEntropyLoss(balanced=True)
        elif task == 'normals':
            loss_dict[task] = nn.ModuleDict()
            loss_task_weights[task] = 10.0
            for p in head_endpoints:
                if p == 'aux':
                    loss_dict[task][p] = criterion.ClusterCrossEntropyLoss(centroids=normals_centroids)
                else:
                    loss_dict[task][p] = criterion.L1Loss(normalize=True)
        elif task == 'depth':
            loss_dict[task] = nn.ModuleDict()
            loss_task_weights[task] = 1.0
            for p in head_endpoints:
                if p == 'aux':
                    loss_dict[task][p] = criterion.BinningCrossEntropyLoss(num_bins=40)
                else:
                    loss_dict[task][p] = criterion.L1Loss()
        elif task == 'edge':
            loss_dict[task] = nn.ModuleDict()
            loss_task_weights[task] = 50.0
            for p in head_endpoints:
                if p == 'aux':
                    loss_dict[task][p] = criterion.CrossEntropyLoss(
                        class_weight=torch.tensor([1. - edge_pos_weight, edge_pos_weight]))
                else:
                    loss_dict[task][p] = criterion.BalancedBinaryCrossEntropyLoss(pos_weight=edge_pos_weight)
        else:
            raise ValueError

    loss = criterion.WeightedSumLoss(tasks, loss_dict, loss_task_weights)
    return loss
