#! /bin/sh

python ./tools/train_cifar100.py \
    --gpus 6 \
    --wd 0.0005 \
    --batch-size 128 \
    --model quick_cnn \
    --resume-from './params/CIFAR100/quick_cnn/baseline/baseline_quickcnn_cifar_57.78.params' \
    --save-name '_icarl_0.5_T=2' \
    --sess-num 9 \
    --lrs 0.1 \
    --lr-decay 0.1 \
    --base-decay-epoch 40 \
    --inc-decay-epoch 100 \
    --epoch 70 \
    --use-pdl \
    --pdl-weight 0.5 \
    --temperature 2 \
    --c-way 5 \
    --k-shot 5 \
    --base-acc 57.78 \
    --dataset 'NC_CIFAR100' \
    --select-best 'select_best2'