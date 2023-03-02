from typing import Any, Callable, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl


def classification_input_format(num_classes: int,
                                preds: torch.Tensor,
                                target: torch.Tensor,
                                threshold=0.5,
                                ignore_index=None):
    if target.shape[1] == 1:
        target = target.squeeze(1)

    if preds.shape[1] == 1:
        preds = preds.squeeze(1)

    if not (len(preds.shape) == len(target.shape) or len(preds.shape) == len(target.shape) + 1):
        raise ValueError(
            "preds and target must have same number of dimensions, or one additional dimension for preds"
        )

    if len(preds.shape) == len(target.shape) + 1:
        # multi class probabilites
        preds = torch.argmax(preds, dim=1)

    valid_mask = (target != ignore_index)
    preds = torch.masked_select(preds, valid_mask)
    target = torch.masked_select(target, valid_mask)

    if len(preds.shape) == len(target.shape) and torch.allclose(preds.float(), preds.int().float()) and num_classes > 1:
        # multi-class
        preds = pl.metrics.utils.to_onehot(preds, num_classes=num_classes)
        target = pl.metrics.utils.to_onehot(target, num_classes=num_classes)

    elif len(preds.shape) == len(target.shape) and preds.dtype == torch.float:
        # binary probablities
        preds = torch.sigmoid(preds)
        preds = (preds >= threshold)

    else:
        raise ValueError

    # transpose class as first dim and reshape
    if len(preds.shape) > 1:
        preds = preds.transpose(1, 0)
        target = target.transpose(1, 0)

    return preds.reshape(num_classes, -1).long(), target.reshape(num_classes, -1).long()


def normalize_tensor(input_tensor, dim):
    norm = torch.norm(input_tensor, p='fro', dim=dim, keepdim=True)
    zero_mask = (norm == 0)
    norm[zero_mask] = 1
    out = input_tensor.div(norm)
    out[zero_mask.expand_as(out)] = 0
    return out


class MeanIoU(pl.metrics.Metric):
    """
    Intersection over union, or Jaccard index calculation.

    Works with binary and multiclass data.
    Accepts logits from a model output or integer class values in prediction.
    Works with multi-dimensional preds and target.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``

    If num_classes == 1, we use the ``self.threshold`` argument.
    This is the case for binary logits.

    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.

    Args:
        num_classes: Specify the number of classes
        ignore_index: Pixels with this label are ignored
        threshold:
            Threshold value for binary or multi-label logits. default: 0.5
        average:
            * `'micro'` computes metric globally
            * `'macro'` computes metric for each class and then takes the mean
        multilabel: If predictions are from multilabel classification.
    """

    def __init__(self,
                 num_classes,
                 ignore_index: int = 255,
                 threshold: float = 0.5,
                 average: str = 'macro',
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None):
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step, process_group=process_group)

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.average = average

        assert self.average in ('micro', 'macro'), \
            "average passed to the function must be either `micro` or `macro`"

        self.add_state("intersection", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("union", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        preds, target = classification_input_format(self.num_classes, preds, target, self.threshold, self.ignore_index)

        self.intersection += torch.sum(preds & target, dim=1)
        self.union += torch.sum(preds | target, dim=1)

    def compute(self):

        if self.average == 'micro':
            mean_iou = self.intersection.sum().float() / self.union.sum()
            mean_iou[mean_iou != mean_iou] = 0
            return mean_iou
        elif self.average == 'macro':
            mean_iou = self.intersection.float() / self.union
            mean_iou[mean_iou != mean_iou] = 0
            return mean_iou.mean()


class MaxF(pl.metrics.Metric):
    """
    Computes maximum f_beta metric 

    For info and references regarding saliency estimation see https://arxiv.org/abs/1805.07567

    Works with binary data. Accepts logits from a model output.
    Works with multi-dimensional preds and target.

    Forward accepts

    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, 1, ...)``
    - ``target`` (float or long tensor): ``(N, ...)`` or ``(N, 1, ...)``

    Args:
        ignore_index: Pixels with this label are ignored
        beta_squared: Beta coefficient squared in the F measure.
        threshold_step:
            Increment for threshold value
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
    """

    def __init__(self,
                 ignore_index: int = 255,
                 beta_squared: float = 0.3,
                 threshold_step: float = 0.05,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None):
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group)
        self.ignore_index = ignore_index
        self.beta_squared = beta_squared

        self.register_buffer("thresholds", torch.arange(threshold_step, 1, threshold_step), persistent=False)

        self.add_state("true_positives", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")
        self.add_state("predicted_positives", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")
        self.add_state("actual_positives", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """

        if target.shape[1] == 1:
            target = target.squeeze(1)

        if preds.shape[1] == 1:
            preds = preds.squeeze(1)

        if len(preds.shape) == len(target.shape) + 1:
            assert preds.shape[1] == 2
            # two class probabilites
            preds = nn.functional.softmax(preds, dim=1)[:, 1, :, :]
        else:
            # squash logits into probabilities
            preds = torch.sigmoid(preds)

        if not len(preds.shape) == len(target.shape):
            raise ValueError("preds and target must have same number of dimensions, or preds one more")

        valid_mask = (target != self.ignore_index)

        for idx, thresh in enumerate(self.thresholds):
            # threshold probablities
            f_preds = (preds >= thresh).long()
            f_target = target.long()

            f_preds = torch.masked_select(f_preds, valid_mask)
            f_target = torch.masked_select(f_target, valid_mask)

            self.true_positives[idx] += torch.sum(f_preds * f_target)
            self.predicted_positives[idx] += torch.sum(f_preds)
            self.actual_positives[idx] += torch.sum(f_target)

    def compute(self):
        """
        Computes F-scores over state and returns the max.
        """
        precision = self.true_positives.float() / self.predicted_positives
        recall = self.true_positives.float() / self.actual_positives

        num = (1 + self.beta_squared) * precision * recall
        denom = self.beta_squared * precision + recall

        # For the rest we need to take care of instances where the denom can be 0
        # for some classes which will produce nans for that class
        fscore = num / denom
        fscore[fscore != fscore] = 0
        return fscore.max()


class MeanErrorInAngle(pl.metrics.Metric):
    """
    Calculates the mean error in the angles of surface normal vectors. For that purpose,
    both prediction and ground truth vectors are normalized before evaluation.

    Args:
        ignore_index: Pixels with this label are ignored
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
    """

    def __init__(self,
                 ignore_index: int = 255,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None):
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         dist_sync_fn=dist_sync_fn)
        self.ignore_index = ignore_index

        self.add_state("sum_deg_diff", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        # pl.metrics.utils._check_same_shape(preds, target)
        valid_mask = (target != self.ignore_index).all(dim=1)

        preds = normalize_tensor(preds, dim=1)
        target = normalize_tensor(target, dim=1)
        deg_diff = torch.rad2deg(2 * torch.atan2(torch.norm(preds - target, dim=1), torch.norm(preds + target, dim=1)))
        deg_diff = torch.masked_select(deg_diff, valid_mask)

        self.sum_deg_diff += torch.sum(deg_diff)
        self.total += deg_diff.numel()

    def compute(self):
        """
        Computes root mean squared error over state.
        """
        return self.sum_deg_diff / self.total


class RMSE(pl.metrics.Metric):
    """
    Computes root mean squared error.

    Args:
        ignore_index: Pixels with this label are ignored
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
    """

    def __init__(self,
                 ignore_index: int = 0,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None):
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         dist_sync_fn=dist_sync_fn)
        self.ignore_index = ignore_index

        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        # pl.metrics.utils._check_same_shape(preds, target)

        valid_mask = (target != self.ignore_index)
        preds = torch.masked_select(preds, valid_mask)
        target = torch.masked_select(target, valid_mask)

        self.sum_squared_error += torch.sum(torch.pow(preds - target, 2))
        self.total += target.numel()

    def compute(self):
        """
        Computes root mean squared error over state.
        """
        return torch.sqrt(self.sum_squared_error / self.total)
