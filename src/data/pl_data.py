import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

from .nyud import NYUD
from .pascal_context import PASCALContext
from . import metrics
from . import transforms


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 num_duplicates: int,
                 dataset_name: str,
                 tasks: str = None,
                 data_dir: str = 'data',
                 batch_size: int = 8,
                 num_workers: int = 4):
        super().__init__()
        assert batch_size % num_duplicates == 0
        self.num_duplicates = num_duplicates
        self.tasks = tasks.split(',')
        self.dataset_cls = globals()[dataset_name]
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dims = self.dataset_cls.image_dims

        self.train_transforms = torchvision.transforms.Compose([
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=self.dims[-2:], cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=self.dims[-2:]),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        self.valid_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=self.dims[-2:]),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        self.edge_pos_weight = self.dataset_cls.edge_pos_weight
        self.task_channel_mapping = {
            'semseg': {
                'final': self.dataset_cls.semseg_num_classes,
                'aux': self.dataset_cls.semseg_num_classes,
            },
            'depth': {
                'final': 1,
                'aux': 40,
            },
            'normals': {
                'final': 3,
                'aux': 40,
            },
            'edge': {
                'final': 1,
                'aux': 2,
            },
            'human_parts': {
                'final': 7,
                'aux': 7,
            },
            'sal': {
                'final': 2,
                'aux': 2,
            }
        }

        self.trainset = self.dataset_cls(self.data_dir,
                                         split='train',
                                         tasks=self.tasks,
                                         transforms=self.train_transforms)
        self.validset = self.dataset_cls(self.data_dir,
                                         split='val',
                                         tasks=self.tasks,
                                         transforms=self.valid_transforms,
                                         retname=True)
        self.normals_centroids = self.trainset.normals_centroids

        self.metrics_dict = nn.ModuleDict({
            'semseg': metrics.MeanIoU(num_classes=self.dataset_cls.semseg_num_classes, compute_on_step=False),
            'depth': metrics.RMSE(compute_on_step=False),
            'normals': metrics.MeanErrorInAngle(compute_on_step=False),
            'human_parts': metrics.MeanIoU(num_classes=7, compute_on_step=False),
            'sal': metrics.MaxF(beta_squared=0.3, compute_on_step=False)
        })

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset,
                                           shuffle=True,
                                           batch_size=self.batch_size // self.num_duplicates,
                                           num_workers=self.num_workers,
                                           pin_memory=True,
                                           drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validset,
                                           batch_size=self.batch_size // self.num_duplicates,
                                           num_workers=self.num_workers,
                                           pin_memory=True)
