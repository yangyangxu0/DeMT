# DeMT

This repo is the official implementation of ["DeMT"](https://arxiv.org/pdf/2103.14030.pdf) as well as the follow-ups. It currently includes code and models for the following tasks:



## Updates

***02/10/2023***


1. Merged Code.

2. Released a series of models. Please look into the [data scaling](https://arxiv.org/abs/2301.03461) paper for more details.

***02/07/2022***

`News`: 

1. The Thirty-Seventh Conference on Artificial Intelligence (AAAI2023) will be held in Washington, DC, USA., from February 7-14, 2023.


***02/01/2022***

1. DeMT got accepted by AAAI 2023. 




## Introduction

**DeMT** (the name `DeMT` stands for **De**formable **M**ixer **T**ransformer for Multi-Task Learning of Dense
Prediction) is initially described in [arxiv](https://arxiv.org/pdf/2301.03461.pdf), which is based on a simple and effective encoder-decoder architecture (i.e., deformable mixer encoder and task-aware transformer decoder). First, the deformable mixer encoder contains two types of operators: the
channel-aware mixing operator leveraged to allow communication among different channels (i.e., efficient channel location mixing), and the spatial-aware deformable operator with deformable convolution applied to efficiently sample more informative spatial locations (i.e., deformed features). Second, the task-aware transformer decoder consists of the task interaction block and task query block. The former is applied to capture task interaction features via self-attention. The latter leverages the deformed features and task-interacted features to generate the corresponding task-specific feature through a query-based Transformer for corresponding task predictions.

DeMT achieves strong performance on PASCAL-Context (`75.33 mIoU semantic segmentation` and `63.11 mIoU Human Segmentation` on test) and
 and NYUD-v2 semantic segmentation (`54.34 mIoU` on test), surpassing previous models by a large margin.

![DeMT](figures/overflow.png)

## Main Results on ImageNet with Pretrained Models

**NYUD-v2 dataset DeMT Models**

| name | pretrain | resolution |acc@1 | acc@5 | #params | FLOPs | FPS| 22K model | 1K model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |:---: |
| Swin-T | ImageNet-1K | 224x224 | 81.2 | 95.5 | 28M | 4.5G | 755 | - | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)/[baidu](https://pan.baidu.com/s/156nWJy4Q28rDlrX-rRbI3w)/[config](configs/swin/swin_tiny_patch4_window7_224.yaml)/[log](https://github.com/SwinTransformer/storage/files/7745562/log_swin_tiny_patch4_window7_224.txt) |
| Swin-S | ImageNet-22K | 224x224 | 83.2 | 97.0 | 50M | 8.7G | 437 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth)/[baidu](https://pan.baidu.com/s/11NC1xdT5BAGBgazdTme5Sg?pwd=swin)/[config](configs/swin/swin_small_patch4_window7_224_22k.yaml) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22kto1k_finetune.pth)/[baidu](https://pan.baidu.com/s/10RFVfjQJhwPfeHrmxQUaLw?pwd=swin)/[config](configs/swin/swin_small_patch4_window7_224_22kto1k_finetune.yaml) |
| Swin-B | ImageNet-22K | 224x224 | 85.2 | 97.5 | 88M | 15.4G | 278 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth)/[baidu](https://pan.baidu.com/s/1y1Ec3UlrKSI8IMtEs-oBXA)/[config](configs/swin/swin_base_patch4_window7_224_22k.yaml) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth)/[baidu](https://pan.baidu.com/s/1n_wNkcbRxVXit8r_KrfAVg)/[config](configs/swin/swin_base_patch4_window7_224_22kto1k_finetune.yaml) |
| Swin-L | ImageNet-22K | 384x384 | 87.3 | 98.2 | 197M | 103.9G | 42 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth)/[baidu](https://pan.baidu.com/s/1sl7o_bJA143OD7UqSLAMoA) | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth)/[baidu](https://pan.baidu.com/s/1X0FLHQyPOC6Kmv2CmgxJvA)/[config](configs/swin/swin_large_patch4_window12_384_22kto1k_finetune.yaml) |


## Citing DeMT multi-task method

```
@inproceedings{xyy2023DeMT,
  title={DeMT: Deformable Mixer Transformer for Multi-Task Learning of Dense Prediction},
  author={Xu, Yangyang and Yang, Yibo and Zhang, Lefei },
  booktitle={Proceedings of the The Thirty-Seventh Conference on Artificial Intelligence (AAAI)},
  year={2023}
}
```


## Getting Started


## Trademarks
