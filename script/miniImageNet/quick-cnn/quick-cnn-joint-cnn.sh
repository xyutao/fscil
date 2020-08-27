#! /bin/sh

python ./tools/train_mini_imagenet.py \
    --gpus 8 \
    --wd 0.0005 \
    --batch-size 128 \
    --model quick_cnn \
    --resume-from './params/MINI_IMAGENET/quick_cnn/baseline/baseline_quickcnn_mini_50.71.params' \
    --save-name '_cum' \
    --sess-num 9 \
    --lrs 0.001 \
    --lr-decay 0.1 \
    --base-decay-epoch 30,45 \
    --inc-decay-epoch 200 \
    --epoch 50 \
    --cum \
    --c-way 5 \
    --k-shot 5 \
    --dataset 'NC_MiniImageNet' \
    --base-acc 50.71 \
    --select-best 'select_best2'
