#! /bin/sh

python ./tools/train_cub200.py \
    --gpus 5 \
    --wd 0.0005 \
    --batch-size 128 \
    --model resnet18 \
    --resume-from './params/CUB200/resnet18_inchead/baseline_resnet18_cub200_68.68.params' \
    --save-name '_AL+fw' \
    --wo-bn \
    --fw \
    --use-ng\
    --ng-var \
    --sess-num 11 \
    --lrs 0.01 \
    --lr-decay 0.1 \
    --base-decay-epoch 40,60 \
    --inc-decay-epoch 110 \
    --epoch 120 \
    --use-AL \
    --AL-weight 350 \
    --fix-conv \
    --fix-epoch 200 \
    --c-way 10 \
    --k-shot 5 \
    --base-acc 68.68 \
    --dataset 'NC_CUB200' \
    --select-best 'select_best2'
