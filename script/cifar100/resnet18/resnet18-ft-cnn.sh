#! /bin/sh

python ./tools/train_cifar100.py \
    --gpus 1 \
    --wd 0.0005 \
    --batch-size 128 \
    --model resnet18 \
    --save-name 'baseline' \
    --resume-from './params/CIFAR100/resnet18/baseline/baseline_resnet18_cifar_64.10.params' \
    --sess-num 9 \
    --lrs 0.1 \
    --lr-decay 0.1 \
    --base-decay-epoch 60,70 \
    --inc-decay-epoch 60 \
    --epoch 70 \
    --c-way 5 \
    --k-shot 5 \
    --base-acc 64.10 \
    --dataset 'NC_CIFAR100' \
    --select-best 'select_best2'
