#! /bin/sh

python ./tools/train_mini_imagenet.py \
    --gpus 7 \
    --wd 0.0005 \
    --batch-size 128 \
    --model quick_cnn \
    --resume-from './params/MINI_IMAGENET/quick_cnn/baseline/baseline_quickcnn_mini_50.71.params' \
    --save-name '_AL_min' \
    --use-center \
    --center-weight 0.1 \
    --center-lr 0.1 \
    --sess-num 9 \
    --lrs 0.1 \
    --lr-decay 0.1 \
    --base-decay-epoch 40 \
    --inc-decay-epoch 90 \
    --epoch 100 \
    --use-ng \
    --ng-var \
    --use-AL \
    --AL-weight 220 \
    --fix-conv \
    --fix-epoch 200 \
    --c-way 5 \
    --k-shot 5 \
    --base-acc 50.71 \
    --dataset 'NC_MiniImageNet' \
    --select-best 'select_best2'
