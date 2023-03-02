import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

from . import builder
from . import utils_model


class MultiTaskModel(pl.LightningModule):
    def __init__(self,
                 model_backbone: str,
                 in_index: str,
                 model_head: str,
                 tasks: list,  # datamodule
                 task_channel_mapping: dict,  # datamodule
                 metrics_dict: nn.ModuleDict,  # datamodule
                 edge_pos_weight: float,  # datamodule
                 normals_centroids: torch.Tensor,  # datamodule
                 iterations: int = 40000,
                 lr: float = 0.01,
                 head_lr_mult: float = 1.0,
                 weight_decay: float = 0.0005,
                 atrc_genotype_path: str = None):
        super().__init__()
        self.tasks = tasks
        self.metrics_dict = metrics_dict
        self.iterations = iterations
        self.lr = lr
        self.head_lr_mult = head_lr_mult
        self.weight_decay = weight_decay

        self.backbone = builder.get_backbone(model_backbone=model_backbone,
                                             pretrained=True)
        self.head = builder.get_head(head_name=model_head,
                                     in_index=in_index,
                                     idx_to_planes=self.backbone.idx_to_planes,
                                     tasks=tasks,
                                     task_channel_mapping=task_channel_mapping,
                                     atrc_genotype_path=atrc_genotype_path)
        self.criterion = builder.get_criterion(tasks=tasks,
                                               head_endpoints=self.head.head_endpoints,
                                               edge_pos_weight=edge_pos_weight,
                                               normals_centroids=normals_centroids)

    def on_fit_start(self):
        if 'edge' in self.tasks:
            self.edge_save_dir = os.path.join(self.logger.log_dir, 'edge_preds')
            if self.trainer.is_global_zero:
                os.makedirs(self.edge_save_dir, exist_ok=True)

    def training_step(self, batch, batch_idx):
        image = batch['image']
        targets = {t: batch[t] for t in self.tasks}

        input_shape = image.shape[-2:]
        features = self.backbone(image)
        out = self.head(features, input_shape, image=image)

        loss, logger_losses = self.criterion(out, targets)

        self.log('train_loss', loss)
        for key, val in logger_losses.items():
            self.log(key + '_train_loss', val)

        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        targets = {t: batch[t] for t in self.tasks}

        input_shape = image.shape[-2:]
        features = self.backbone(image)
        out = self.head(features, input_shape, image=image) #feature(list) [1,128,107,140],[1,256,54,70], [1,512,27,35],[1,1024,14,18]

        for task in self.tasks:
            task_target = targets[task]
            task_pred = out['final'][task]

            if task == 'depth':
                # threshold negative values
                task_pred.clamp_(min=0.)

            if task == 'edge':
                # edge predictions are saved for later evaluation
                utils_model.save_predictions('edge', task_pred, batch['meta'], self.edge_save_dir)
            else:
                self.metrics_dict[task](task_pred, task_target)

    def validation_epoch_end(self, outputs):
        metrics_val = {}
        for task in self.tasks:
            if task == 'edge':
                continue
            metrics_val[task] = self.metrics_dict[task].compute()
            self.log('_'.join(
                [task, 'valid', self.metrics_dict[task].__class__.__name__]), metrics_val[task], sync_dist=True)

    test_step = validation_step
    test_epoch_end = validation_epoch_end

    def configure_optimizers(self):
        def get_polynomial_lambda(begin_lr=0.0001, decay_steps=100000, end_lr=0., power=0.9):
            def polynomial_lr(step_now: int) -> float:
                return (begin_lr - end_lr) * ((1 - step_now / decay_steps) ** power) + end_lr

            return polynomial_lr

        params = self._get_parameters()
        optimizer = torch.optim.SGD(
            lr=self.lr, momentum=0.9, weight_decay=self.weight_decay, params=params)
        scheduler = {
            'scheduler': utils_model.PolynomialLR(optimizer, self.iterations, gamma=0.9, min_lr=0),
            'interval': 'step'
        }
        return [optimizer], [scheduler]

    def _get_parameters(self):
        backbone_params = []
        head_params = []
        params_dict = dict(self.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' not in key:
                head_params.append(value)
            else:
                backbone_params.append(value)
        params = [{'params': backbone_params, 'lr': self.lr},
                  {'params': head_params, 'lr': self.lr * self.head_lr_mult}]
        return params
