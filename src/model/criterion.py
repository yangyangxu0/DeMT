import math
import torch
import torch.nn as nn


class WeightedSumLoss(nn.Module):
    """
    Simple multi-task loss, consisting of a weighted sum of individual task losses.
    """
    def __init__(self, tasks, loss_dict, loss_task_weights):
        super().__init__()
        self.tasks = tasks
        self.loss_dict = loss_dict
        self.loss_task_weights = loss_task_weights

    def forward(self, out, lab):
        losses = []
        logger_losses = {}

        for p in out:
            for t in out[p]:
                loss = self.loss_task_weights[t] * self.loss_dict[t][p](out[p][t], lab[t])
                logger_losses[t + '_' + p] = loss.detach()
                losses.append(loss)

        tot_loss = sum(losses)
        return tot_loss, logger_losses


class ClusterCrossEntropyLoss(nn.Module):

    def __init__(self, centroids, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer('centroids', centroids)

    def forward(self, out, label, reduction='mean'):
        N, _, H, W = label.shape
        ignore_mask = (label == self.ignore_index).any(dim=1)
        label = label.permute(0, 2, 3, 1).view(N, H * W, 3)
        centroid_idx = torch.argmin(torch.cdist(
            label, self.centroids), dim=2).view(N, H, W).long()
        centroid_idx[ignore_mask] = self.ignore_index
        loss = nn.functional.cross_entropy(
            out, centroid_idx, ignore_index=self.ignore_index, reduction=reduction)
        return loss


class BinningCrossEntropyLoss(nn.Module):

    def __init__(self, num_bins, alpha=0.7, beta=10.0, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta
        self.discretization = 'log'
        self.num_bins = num_bins

    def _create_disc_label(self, label):
        if self.discretization == 'log':
            # assert (label > 0).all()
            disc_label = torch.floor(
                self.num_bins * torch.log(label / self.alpha) / math.log(self.beta / self.alpha))
        else:  # uniform
            disc_label = torch.floor(
                self.num_bins * (label - self.alpha) / (self.beta - self.alpha))
        disc_label[disc_label < 0] = 0
        # ignore_regions get readded anyways
        disc_label[disc_label > self.num_bins - 1] = self.num_bins - 1
        return disc_label.long()

    def forward(self, out, label, reduction='mean'):

        ignore_mask = (label == self.ignore_index)
        disc_label = self._create_disc_label(label)
        disc_label[ignore_mask] = self.ignore_index
        disc_label = disc_label.squeeze(1)
        loss = nn.functional.cross_entropy(
            out, disc_label, ignore_index=self.ignore_index, reduction=reduction)
        return loss


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss with ignore regions.
    """
    def __init__(self, ignore_index=255, class_weight=None, balanced=False):
        super().__init__()
        self.ignore_index = ignore_index
        if balanced:
            assert class_weight is None
        self.balanced = balanced
        if class_weight is not None:
            self.register_buffer('class_weight', class_weight)
        else:
            self.class_weight = None

    def forward(self, out, label, reduction='mean'):
        label = torch.squeeze(label, dim=1).long()
        if self.balanced:
            mask = (label != self.ignore_index)
            masked_label = torch.masked_select(label, mask)
            assert torch.max(masked_label) < 2  # binary
            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            w_pos = num_labels_neg / num_total
            class_weight = torch.stack((1. - w_pos, w_pos), dim=0)
            loss = nn.functional.cross_entropy(
                out, label, weight=class_weight, ignore_index=self.ignore_index, reduction='none')
        else:
            loss = nn.functional.cross_entropy(out,
                                               label,
                                               weight=self.class_weight,
                                               ignore_index=self.ignore_index,
                                               reduction='none')
        if reduction == 'mean':
            n_valid = (label != self.ignore_index).sum()
            return loss.sum() / max(n_valid, 1)
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss


class BalancedBinaryCrossEntropyLoss(nn.Module):
    """
    Balanced binary cross entropy loss with ignore regions.
    """
    def __init__(self, pos_weight=None, ignore_index=255):
        super().__init__()
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, output, label, reduction='mean'):

        mask = (label != self.ignore_index)
        masked_label = torch.masked_select(label, mask)
        masked_output = torch.masked_select(output, mask)

        # weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            w = num_labels_neg / num_total
            if w == 1.0:
                return 0
        else:
            w = torch.as_tensor(self.pos_weight, device=output.device)
        factor = 1. / (1 - w)

        loss = nn.functional.binary_cross_entropy_with_logits(
            masked_output,
            masked_label,
            pos_weight=w*factor,
            reduction=reduction)
        loss /= factor
        return loss


class L1Loss(nn.Module):
    """
    L1 loss with ignore regions.
    normalize: normalization for surface normals
    """
    def __init__(self, normalize=False, ignore_index=255):
        super().__init__()
        self.normalize = normalize
        self.ignore_index = ignore_index

    def forward(self, out, label, reduction='mean'):

        if self.normalize:
            out = nn.functional.normalize(out, p=2, dim=1)

        mask = (label != self.ignore_index).all(dim=1, keepdim=True)
        n_valid = torch.sum(mask).item()
        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        if reduction == 'mean':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='sum') / max(n_valid, 1)
        elif reduction == 'sum':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='sum')
        elif reduction == 'none':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='none')
