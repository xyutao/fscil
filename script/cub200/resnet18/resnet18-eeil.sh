#! /bin/sh

python ./tools/train_cub200.py \
    --gpus 7 \
    --wd 0.0005 \
    --batch-size 64 \
    --model resnet18 \
    --resume-from './params/CUB200/resnet18_inchead/baseline_resnet18_cub200_68.68.params' \
    --save-name 'cub200_EEIL' \
    --sess-num 11 \
    --lrs 0.01 \
    --lr-decay 0.1 \
    --base-decay-epoch 27,36 \
    --inc-decay-epoch 200 \
    --epoch 100 \
    --fix-conv \
    --fix-epoch 125 \
    --use-pdl \
    --pdl-weight 0.8 \
    --temperature 2 \
    --use-oce \
    --oce-weight 0.004\
    --c-way 5 \
    --k-shot 5 \
    --base-acc 68.68 \
    --dataset 'NC_CUB200' \
    --select-best 'select_best2'