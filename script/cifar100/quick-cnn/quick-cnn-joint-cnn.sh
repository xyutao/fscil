#! /bin/sh

python ./tools/train_cifar100.py \
    --gpus 0 \
    --wd 0.0005 \
    --batch-size 128 \
    --model quick_cnn \
    --resume-from './params/CIFAR100/quick_cnn/baseline/baseline_quickcnn_cifar_57.78.params' \
    --save-name '_baseline_quick' \
    --sess-num 1 \
    --lrs 0.0001 \
    --lr-decay 0.1 \
    --base-decay-epoch 50,80 \
    --inc-decay-epoch 40\
    --epoch 100 \
    --cum \
    --c-way 5 \
    --k-shot 5 \
    --dataset 'NC_CIFAR100' \
    --select-best 'select_best2'
