# FQSR

Official codes for ACMMM 2021 paper "Fully Quantized Image Super-Resolution Networks".

https://arxiv.org/abs/2011.14265

![](/figs/visualization.png)

## Installation

Python 3.6.5

torch 1.2.0

torchvision 0.4.0

For more requirements, please refer to requirements.txt

## Data Preparation

Followed the existing SR settings, the models are trained on [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset and it can be downloaded from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (7.1GB).

For model evaluation, the [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) (250MB) can be downloaded from the hyper-link provided. It includes Set5, Set14, B100 and Urban100. We also adopted the DIV2K photo id from 801 to 900 for DIV2K dataset evaluation.

The data path can be changed in train.sh

## Model Training

For full-precision SRResNet model training,

```commandline
CUDA_VISIBLE_DEVICES=0 bash train.sh config/config.baseline.train-scratch.div2k.fp.srresnet $name
```

For FQSR model training (SRResNet-based),

```commandline
CUDA_VISIBLE_DEVICES=0 bash train.sh config/config.lsq.finetune.div2k.bit.srresnet $name
```

In order to receive the most promising results, the resume from full-precision can be specified in the corresponding `config/config.lsq.finetune.div2k.bit.srresnet` file.

Similarly, for full-precision EDSR model training,

```commandline
CUDA_VISIBLE_DEVICES=0 bash train.sh config/config.baseline.train-scratch.div2k.fp.edsr $name
```

For FQSR model training (EDSR-based),

```commandline
CUDA_VISIBLE_DEVICES=0 bash train.sh config/config.lsq.finetune.div2k.bit.edsr $name
```

## Model Evaluation

For model evaluation, the resume path of the tested model can be specified in the corresponding `config/config.lsq.finetune.div2k.bit.srresnet` file.
Remember to turn on the `--test_only` option.

## Citation

If this respository is useful for your research, please consider citing:

```angular2html
@article{wang2021fully,
  title={Fully Quantized Image Super-Resolution Networks},
  author={Hu Wang, Peng Chen, Bohan Zhuang, Chunhua Shen},
  journal={ACM Multimedia},
  year={2021}
}
```

## Acknowledgement
Part of the code is revised from the [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch) and [SRResNet](https://github.com/twtygqyy/pytorch-SRResNet).
