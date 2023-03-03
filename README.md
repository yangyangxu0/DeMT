# DeMT

This repo is the official implementation of ["DeMT"](https://arxiv.org/abs/2301.03461) as well as the follow-ups. It currently includes code and models for the following tasks:



## Updates

***02/10/2023***
1. We will release the code of DeMT at the end of February.

2. Merged Code.

3. Released a series of models. Please look into the [data scaling](https://arxiv.org/abs/2301.03461) paper for more details.

***02/07/2023***

`News`: 

1. The Thirty-Seventh Conference on Artificial Intelligence (AAAI2023) will be held in Washington, DC, USA., from February 7-14, 2023.


***02/01/2023***

1. DeMT got accepted by AAAI 2023. 


## Introduction

**DeMT** (the name `DeMT` stands for **De**formable **M**ixer **T**ransformer for Multi-Task Learning of Dense
Prediction) is initially described in [arxiv](https://arxiv.org/pdf/2301.03461.pdf), which is based on a simple and effective encoder-decoder architecture (i.e., deformable mixer encoder and task-aware transformer decoder). First, the deformable mixer encoder contains two types of operators: the
channel-aware mixing operator leveraged to allow communication among different channels (i.e., efficient channel location mixing), and the spatial-aware deformable operator with deformable convolution applied to efficiently sample more informative spatial locations (i.e., deformed features). Second, the task-aware transformer decoder consists of the task interaction block and task query block. The former is applied to capture task interaction features via self-attention. The latter leverages the deformed features and task-interacted features to generate the corresponding task-specific feature through a query-based Transformer for corresponding task predictions.

DeMT achieves strong performance on PASCAL-Context (`75.33 mIoU semantic segmentation` and `63.11 mIoU Human Segmentation` on test) and
 and NYUD-v2 semantic segmentation (`54.34 mIoU` on test), surpassing previous models by a large margin.

![DeMT](figures/overflow.png)

## Main Results on ImageNet with Pretrained Models

**DeMT on NYUD-v2 dataset**

| model|backbone|#params| FLOPs | SemSeg| Depth | Noemal|Boundary| model checkpopint | log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |:---: |
| DeMT |HRNet-18| 4.76M  | 22.07G  | 39.18 | 0.5922 | 20.21| 76.4 | [Google Drive]() | [log]()  |
| DeMT | Swin-T | 32.07M | 100.70G | 46.36 | 0.5871 | 20.60| 76.9 | [Google Drive](https://drive.google.com/file/d/1IfQRVyvaVkEfybzh4QAz9Vq_0U38Hngq/view?usp=share_link) | [log](https://drive.google.com/file/d/1eAtQVJLcvIOMwAfKyl2NmYfe3hPne_WK/view?usp=share_link)  |
| DeMTxdepth2 | Swin-T | 36.6M| - | 47.45 | 0.5563| 19.90| 77.0 | [Google Drive](https://drive.google.com/file/d/1Rz4R9vu8bGtskpJDlVfgexYZoHtz8j8k/view?usp=share_link) | [log](https://drive.google.com/file/d/1TPo4pMjbhPAn3gxKOt4P7hVSPJe1Lpsn/view?usp=share_link)  |
| DeMT | Swin-S | 53.03M | 121.05G | 51.5 | 0.5474 | 20.02 | 78.1 | [Google Drive](https://drive.google.com/drive/folders/1jINF9WOyILqrPcsprWbM5VSCEWozsc1c) | [log](https://drive.google.com/drive/folders/1jINF9WOyILqrPcsprWbM5VSCEWozsc1c)|
| DeMT | Swin-B | 90.9M | 153.65G | 54.34 | 0.5209 | 19.21 | 78.5 | [Google Drive]() | [log]() |
| DeMT | Swin-L | 201.64M | -G | 56.94 | 0.5007 | 19.14 | 78.8 | [Google Drive]() | [log]() |

**DeMT on PASCAL-Contex dataset**

| model | backbone |  SemSeg | PartSeg | Sal | Normal| Boundary| 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DeMT |HRNet-18| 59.23 | 57.93 | 83.93| 14.02 | 69.80 |
| DeMT | Swin-T | 69.71 | 57.18 | 82.63| 14.56 | 71.20 |
| DeMT | Swin-S | 72.01 | 58.96 | 83.20| 14.57 | 72.10 | 
| DeMT | Swin-B | 75.33 | 63.11 | 83.42| 14.54 | 73.20 |



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
**Install**

```
conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.1 -c pytorch
conda install pytorch-lightning==1.1.8 -c conda-forge
conda install opencv==4.4.0 -c conda-forge
conda install scikit-image==0.17.2
```

**Data Prepare**
```
wget https://data.vision.ee.ethz.ch/brdavid/atrc/NYUDv2.tar.gz
wget https://data.vision.ee.ethz.ch/brdavid/atrc/PASCALContext.tar.gz
tar xfvz ./NYUDv2.tar.gz 
tar xfvz ./PASCALContext.tar.gz
```


**Train**

To train DeMT model:
```
python ./src/main.py --cfg ./config/t-nyud/swin/siwn_t_DeMT.yaml --datamodule.data_dir $DATA_DIR --trainer.gpus 8
```

**Evaluation**

- When the training is finished, the boundary predictions are saved in the following directory: ./logger/NYUD_xxx/version_x/edge_preds/ .
- The evaluation of boundary detection use the MATLAB-based [SEISM](https://github.com/jponttuset/seism) repository to obtain the optimal-dataset-scale-F-measure (odsF) scores.


## Acknowledgement
This repository is based [ATRC](https://github.com/brdav/atrc). Thanks to [ATRC](https://github.com/brdav/atrc)!

