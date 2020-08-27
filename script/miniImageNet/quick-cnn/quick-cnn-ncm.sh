#! /bin/sh

python ./tools/train_mini_imagenet.py \
    --gpus 8 \
    --wd 0.0005 \
    --batch-size 128 \
    --model quick_cnn \
    --resume-from './params/MINI_IMAGENET/quick_cnn/baseline/baseline_quickcnn_mini_50.71.params' \
    --save-name '_NME' \
    --sess-num 9 \
    --lrs 0.1 \
    --lr-decay 0.1 \
    --base-decay-epoch 40 \
    --inc-decay-epoch 90 \
    --epoch 100 \
    --use-nme \
    --nme-weight 1 \
    --fix-conv \
    --fix-epoch 100 \
    --c-way 5 \
    --k-shot 5 \
    --base-acc 50.71 \
    --dataset 'NC_MiniImageNet' \
    --select-best 'select_best2'
