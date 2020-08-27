#! /bin/sh

python ./tools/train_cub200.py \
    --gpus 9 \
    --wd 0.0005 \
    --batch-size 128 \
    --model resnet18 \
    --resume-from './params/CUB200/resnet18_inchead/baseline_resnet18_cub200_68.68.params' \
    --save-name 'cub200_icarl' \
    --sess-num 11 \
    --lrs 0.01 \
    --lr-decay 0.1 \
    --base-decay-epoch 27,36 \
    --inc-decay-epoch 200 \
    --epoch 100 \
    --use-pdl \
    --pdl-weight 1 \
    --fix-conv \
    --fix-epoch 125 \
    --temperature 2 \
    --c-way 5 \
    --k-shot 5 \
    --base-acc 68.68 \
    --dataset 'NC_CUB200' \
    --select-best 'select_best2'