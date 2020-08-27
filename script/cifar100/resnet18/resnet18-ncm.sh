#! /bin/sh

python ./tools/train_cifar100_for_nc.py \
    --gpus 1 \
    --wd 0.0005 \
    --batch-size 128 \
    --model resnet18 \
    --resume-from './params/CIFAR100/resnet18/baseline/baseline_resnet18_cifar_64.10.params' \
    --save-name 'nme' \
    --sess-num 9 \
    --lrs 0.1 \
    --lr-decay 0.1 \
    --base-decay-epoch 40 \
    --inc-decay-epoch 60 \
    --epoch 70 \
    --use-nme \
    --nme-weight 50 \
    --fix-conv \
    --fix-epoch 100 \
    --c-way 5 \
    --k-shot 5 \
    --fw \
    --base-acc 64.10 \
    --dataset 'NC_CIFAR100' \
    --select-best 'select_best2'
