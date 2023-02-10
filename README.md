# Coming soon
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

![teaser](figures/teaser.png)

## Main Results on ImageNet with Pretrained Models

**ImageNet-1K and ImageNet-22K Pretrained Swin-V1 Models**

| name | backbone |  | #params | FLOPs | SemSeg |Depth | Normal| Bound| 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |:---: |
| DeMT | Swin-T |  |  |  |  |  |  | - | [github]|

## Citing Swin Transformer

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
