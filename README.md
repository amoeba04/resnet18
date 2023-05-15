# CIFAR-10 Classification using ResNet18

## Train
- ResNet-18 training on CIFAR-10 with default settings:
```
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --batch_size 128 --epochs 50 --augment_type '0' --lr 1e-2 --lr_min 1e-4 --lr_scale '0' --lr_sched_type '0' --label_smoothing 0.0 --weight_decay 0.0 --optimizer_type '0' --wandb
```
- ResNet-18 training on CIFAR-10 with different settings:
```
# Use RandAug, LR Scaling, Cosine LR Scheduler, Label Smoothing, Weight Decay, AdamW Optimizer
CUDA_VISIBLE_DEVICES=0 python main.py --seed 0 --batch_size 128 --epochs 100 --augment_type '3' --lr 1e-4 --lr_min 1e-6 --lr_scale '1' --lr_sched_type '1' --label_smoothing 0.2 --weight_decay 0.05 --optimizer_type '2' --wandb
```

## Inference
You can just edit '--model_num' to ensemble models. Inference was conducted by looping through the seed numbers.
```
# Ensemble 2 models trained with following settings
CUDA_VISIBLE_DEVICES=0 python ensemble.py --model_num 2 --batch_size 128 --epochs 3 --augment_type '0' --lr 1e-2 --lr_min 1e-4 --lr_scale '0' --lr_sched_type '0' --label_smoothing 0.0 --weight_decay 0.0 --optimizer_type '0'
```

## Environments
NVIDIA DGX A100  
CUDA 11.7, pytorch 2.0, torchvision 0.15.1, timm 0.6.12

## Acknowledgement
This repository is based on the repository https://github.com/heechul-knu/cifar-baseline.