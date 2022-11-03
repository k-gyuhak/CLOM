# CLOM

This is the official repository of Continual Learning Based on OOD Detection and Task Masking ([CLOM](https://arxiv.org/abs/2203.09450)) (CVPRW 2022)

# Note
Check out our related papers.
- A Theoretical Study on Solving Continual Learning, NeurIPS 2022  [[PDF](url)]. It provides a theoretical analysis/justification/generalization of any CIL methods including CLOM, and it also shows necessary and sufficient conditions for good CIL performances.
- A Multi-Head Model for Continual Learning via Out-of-Distribution Replay, CoLLAs 2022 [[PDF](https://arxiv.org/abs/2208.09734)]. It is a novel replay method based on OOD detection. The proposed method requires little exemplars and can perform OOD detection in the continual learning setting.

# Environments

The code has been tested on two different machines with

1. 2x GTX 1080
- cuda=10.2
- pytorch=1.6.0
- torchvision=0.7.0
- cudatoolkit=10.2.89
- tensorboardx=2.1
- apex=0.1
- diffdist=0.1
- gdown=4.4.0

2. 1x RTX 3090
- cuda=11.4
- pytorch=1.7.1
- torchvision=0.8.2
- cudatoolkit=11.0.221
- tensorboardx=2.1
- diffdist=0.1
- gdown=4.4.0

Please install the necessary packages

# Training
Please run train_DATASET.sh for a single gpu machine or train_DATASET_multigpu.sh for multi gpu. e.g. 
```
bash train_cifar10.sh
```
or 
```
bash train_cifar10_multigpu.sh
```
For mixed precision, use --amp

# Evaluation using pre-trained models
Please download the pre-trained models and calibration parameters by running download_pretrained_models.py or download manually from [link](https://drive.google.com/drive/folders/1182VgriR841mvW2LXiARTSoBvbPho4Pf). The models and calibration parameters need to be saved under ./logs/DATASET/linear_task_TASK_ID, where DATASET are one of [mnist, cifar10, cifar100_10t, cifar100_20t, tinyImagenet_5t, tinyImageNet_10t] and TASK_ID is the last task id in the experiment (e.g. 9 for cifar100_10t).

For CIL of exemplar-free method CLOM(-c), run the following line
```
python eval.py --mode cil --dataset cifar10 --model resnet18 --cil_task 4 --printfn 'cil.txt' --all_dataset --disable_cal
```

For CIL of memory buffer method CLOM, run the following line
```
python eval.py --mode cil --dataset cifar10 --model resnet18 --cil_task 4 --printfn 'cil.txt' --all_dataset
```

For TIL, run the following line
```
python eval.py --mode test_marginalized_acc --dataset cifar10 --model resnet18 --t 4 --all_dataset --printfn 'til.txt'
```

You may change --dataset, --cil_task for other experiments

# Results
The provided pre-trained models give the following results

CIL
|          | MNIST |  CIFAR10 | CIFAR100-10T | CIFAR100-20t | T-ImageNet-5T | T-ImageNet-10T |
| ---------| ------| ----- | ----- | ----- | ----- | ----- |
| CLOM(-c) | 94.73 | 88.75 | 62.82 | 54.74 | 45.74 | 47.40 |
| CLOM     | 96.50 | 88.62 | 65.21 | 58.14 | 52.53 | 47.76 |


TIL
|          | MNIST |  CIFAR10 | CIFAR100-10T | CIFAR100-20t | T-ImageNet-5T | T-ImageNet-10T |
| ---------| ------| ----- | ----- | ----- | ----- | ----- |
| CLOM(-c) | 99.92 | 98.66 | 91.88 | 94.41 | 68.40 | 72.20 |

CLOM and CLOM(-c) are the same as calibration does not affect TIL performance.

# Cite
If you found our paper useful, please cite it!
```bibtex
@InProceedings{Kim_2022_CVPR,
    author    = {Kim, Gyuhak and Esmaeilpour, Sepideh and Xiao, Changnan and Liu, Bing},
    title     = {Continual Learning Based on OOD Detection and Task Masking},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {3856-3866}
}
```

# Acknowledgement
The code uses the source code from [CSI](https://github.com/alinlab/CSI) and [HAT](https://github.com/joansj/hat).
