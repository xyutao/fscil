#! /bin/sh

python ./tools/train_mini_imagenet.py \
    --gpus 2 \
    --wd 0.0005 \
    --batch-size 128 \
    --model resnet18 \
    --resume-from './params/MINI_IMAGENET/resnet18/baseline/baseline_resnet18_mini_61.31.params' \
    --save-name '_cvpr19' \
    --sess-num 9 \
    --lrs 0.1 \
    --lr-decay 0.1 \
    --base-decay-epoch 40,60 \
    --inc-decay-epoch 200 \
    --epoch 100 \
    --use-nme \
    --nme-weight 1 \
    --fix-conv \
    --fix-epoch 100 \
    --c-way 5 \
    --k-shot 5 \
    --base-acc 61.31 \
    --dataset 'NC_MINI_IMAGENET' \
    --select-best 'select_best2'