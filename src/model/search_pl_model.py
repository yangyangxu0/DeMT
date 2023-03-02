import os
import json
import torch
import torch.nn as nn
import pytorch_lightning as pl

from . import builder
from . import utils_model


class MultiTaskSearchModel(pl.LightningModule):
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
                 gumbel_temp: float = 1.0,
                 entropy_weight: float = 0.0):
        super().__init__()
        self.tasks = tasks
        self.metrics_dict = metrics_dict
        self.iterations = iterations
        self.lr = lr
        self.head_lr_mult = head_lr_mult
        self.weight_decay = weight_decay
        self.gumbel_temp = gumbel_temp
        self.entropy_weight = entropy_weight
        self.automatic_optimization = False

        self.backbone = builder.get_backbone(model_backbone=model_backbone,
                                             pretrained=True)
        self.head = builder.get_head(head_name=model_head,
                                     in_index=in_index,
                                     idx_to_planes=self.backbone.idx_to_planes,
                                     tasks=tasks,
                                     task_channel_mapping=task_channel_mapping)
        self.criterion = builder.get_criterion(tasks=tasks,
                                               head_endpoints=self.head.head_endpoints,
                                               edge_pos_weight=edge_pos_weight,
                                               normals_centroids=normals_centroids)

    def on_fit_start(self):
        if 'edge' in self.tasks:
            self.edge_save_dir = os.path.join(self.logger.log_dir, 'edge_preds')
            if self.trainer.is_global_zero:
                os.makedirs(self.edge_save_dir, exist_ok=True)

    def training_step(self, batch, batch_idx, optimizer_idx):
        (opt_weights, opt_arch) = self.optimizers(use_pl_optimizer=True)

        image = batch['image']
        targets = {t: batch[t] for t in self.tasks}
        input_shape = image.shape[-2:]
        features = self.backbone(image)

        out = self.head(features, input_shape, image=image,
                        gumbel_temp=self.gumbel_temp, mode='gumbel')
        loss, logger_losses = self.criterion(out, targets)

        # add regularization loss
        loss += self.entropy_weight * self.entropy_regularizer()

        # we can set optimizer=None here with the appropriate backend
        self.manual_backward(loss, None)

        # for pytorch-lightning 1.1.8, zero_grad() is called automatically
        opt_weights.step()
        opt_arch.step()

        self.log('train_loss', loss)
        for key, val in logger_losses.items():
            self.log(key + '_train_loss', val)
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        targets = {t: batch[t] for t in self.tasks}

        input_shape = image.shape[-2:]
        features = self.backbone(image)
        out = self.head(features, input_shape, image=image,
                        gumbel_temp=self.gumbel_temp, mode='argmax')

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

        self.log('entropy_weight', self.entropy_weight, sync_dist=True)

        if self.trainer.is_global_zero:
            atrc_genotype = {t: {} for t in self.tasks}
            for t in self.tasks:
                for s in self.tasks:
                    atrc_genotype[t][s] = torch.argmax(self.head.atrc_module.cp_blocks[t][s].arch_param.data, dim=-1).item()
            with open(os.path.join(self.logger.log_dir, 'atrc_genotype.json'), 'w') as f:
                json.dump({'data': atrc_genotype, 'step': self.global_step}, f)

    test_step = validation_step
    test_epoch_end = validation_epoch_end

    def configure_optimizers(self):
        optimizer_weights = torch.optim.SGD(lr=self.lr,
                                            momentum=0.9,
                                            weight_decay=self.weight_decay,
                                            params=self.weight_parameters())
        optimizer_arch = torch.optim.Adam(
            lr=0.0005, weight_decay=0, params=self.arch_parameters())
        scheduler = {
            'scheduler': utils_model.PolynomialLR(optimizer_weights,
                                                  self.iterations,
                                                  gamma=0.9,
                                                  min_lr=0),
            'interval': 'step'
        }
        return [optimizer_weights, optimizer_arch], [scheduler]

    def weight_parameters(self):
        backbone_params = []
        head_params = []
        params_dict = dict(self.named_parameters())
        for key, value in params_dict.items():
            if not key.endswith('arch_param'):
                if 'backbone' not in key:
                    head_params.append(value)
                else:
                    backbone_params.append(value)
        params = [{'params': backbone_params, 'lr': self.lr},
                  {'params': head_params, 'lr': self.lr * self.head_lr_mult}]
        return params

    def arch_parameters(self):
        for key, value in self.named_parameters():
            if key.endswith('arch_param'):
                yield value

    def entropy_regularizer(self):
        entropies = []
        for t in self.tasks:
            for s in self.tasks:
                params = self.head.atrc_module.cp_blocks[t][s].arch_param
                entropies.append(-(params.softmax(dim=-1) *
                                 params.log_softmax(dim=-1)).sum())
        # take the mean, making it independent of the number of selection sites
        return torch.mean(torch.stack(entropies, dim=0))
