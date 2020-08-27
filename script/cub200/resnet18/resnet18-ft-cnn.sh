#! /bin/sh

python ./tools/train_cub200.py \
    --gpus 6 \
    --wd 0.0005 \
    --batch-size 128 \
    --model resnet18 \
    --resume-from './params/CUB200/resnet18_inchead/baseline_resnet18_cub200_68.68.params' \
    --save-name 'ft-cnn' \
    --sess-num 11 \
    --lrs 0.1 \
    --lr-decay 0.1 \
    --base-decay-epoch 60,100 \
    --inc-decay-epoch 200 \
    --epoch 100 \
    --c-way 10 \
    --k-shot 5 \
    --base-acc 68.68 \
    --dataset 'NC_CUB200' \
    --select-best 'select_best2'