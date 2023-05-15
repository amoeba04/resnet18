# CIFAR-10 Classification using ResNet18

- train
```
./run.sh
```
or
```
CUDA_VISIBLE_DEVICES=3 python main.py --seed 0 --batch_size 128 --epochs 50 --augment_type 'augmix' --lr 1e-2 --lr_min 1e-4 --lr_scale --lr_sched_type 'cosine' --label_smoothing 0.2 --weight_decay 0.05 --optimizer_type 'adamw'
```

- inference
```
python ensemble.py
```
