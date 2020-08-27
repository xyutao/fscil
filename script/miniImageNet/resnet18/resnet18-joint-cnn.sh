#! /bin/sh

python ./tools/train_mini_imagenet.py \
    --gpus 1 \
    --wd 0.0005 \
    --batch-size 128 \
    --model resnet18 \
    --resume-from './params/MINI_IMAGENET/resnet18/baseline/baseline_resnet18_mini_61.31.params' \
    --save-name '_cum_lr1e-3' \
    --sess-num 9 \
    --lrs 0.1 \
    --lr-decay 0.1 \
    --base-decay-epoch 40,60 \
    --inc-decay-epoch 45 \
    --epoch 50 \
    --cum \
    --c-way 5 \
    --k-shot 5 \
    --dataset 'NC_MiniImageNet' \
    --select-best 'select_best2'
