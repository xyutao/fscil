#! /bin/sh

python ./tools/train_cifar100.py \
    --gpus 6 \
    --wd 0.0005 \
    --batch-size 128 \
    --model quick_cnn \
    --resume-from './params/CIFAR100/quick_cnn/baseline/baseline_quickcnn_cifar_57.78.params' \
    --save-name '_plc_quick' \
    --sess-num 9 \
    --lrs 0.1 \
    --lr-decay 0.1 \
    --base-decay-epoch 40 \
    --inc-decay-epoch 200 \
    --epoch 50 \
    --use-AL \
    --use-ng \
    --fw \
    --c-way 5 \
    --k-shot 5 \
    --base-acc 57.78\
    --dataset 'NC_CIFAR100' \
    --select-best 'select_best2'
