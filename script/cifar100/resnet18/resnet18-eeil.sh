#! /bin/sh

python ./tools/train_cifar100_for_nc.py \
    --gpus 5 \
    --wd 0.0005 \
    --batch-size 128 \
    --model resnet18 \
    --resume-from './params/CIFAR100/resnet18/baseline/baseline_resnet18_cifar_64.10.params' \
    --save-name 'EEIL' \
    --sess-num 9 \
    --lrs 0.1 \
    --lr-decay 0.1 \
    --base-decay-epoch 27,36 \
    --inc-decay-epoch 60 \
    --epoch 70 \
    --fix-conv \
    --fix-epoch 100 \
    --use-pdl \
    --pdl-weight 1 \
    --temperature 2 \
    --use-oce \
    --oce-weight 0.004\
    --c-way 5 \
    --k-shot 5 \
    --base-acc 64.10 \
    --dataset 'NC_CIFAR100' \
    --select-best 'select_best2'