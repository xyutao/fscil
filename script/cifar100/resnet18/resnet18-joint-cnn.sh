#! /bin/sh

python ./tools/train_cifar100_for_nc.py \
    --gpus 1 \
    --wd 0.0005 \
    --batch-size 128 \
    --model resnet18 \
    --resume-from './params/CIFAR100/resnet18/baseline/baseline_resnet18_cifar_64.10.params' \
    --save-name 'cum' \
    --sess-num 9 \
    --lrs 0.1 \
    --lr-decay 0.1 \
    --base-decay-epoch 80,120 \
    --inc-decay-epoch 60 \
    --cum \
    --epoch 70 \
    --c-way 5 \
    --k-shot 5 \
    --dataset 'NC_CIFAR100' \
    --base-acc 64.10 \
    --select-best 'select_best2'
