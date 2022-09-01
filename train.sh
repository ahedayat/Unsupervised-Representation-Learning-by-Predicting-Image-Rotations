
# Hardware
num_workers=2

# Architecture
backbone="resnet34"

# Data Path
train_data_path="/content/datasets/cifar10"
val_data_path="/content/datasets/cifar10"

# Optimizer Parameters
optimizer="adam"
num_epochs=20
batch_size=1024

# Learning Rate 
lr=1e-3
momentum=0.9
weight_decay=5e-3

# Checkpoints Parameters
ckpt_save_path="./checkpoints"
# ckpt_load_path="./checkpoints"
ckpt_prefix="rotation_"
ckpt_save_freq=5

# Report
report_path="./reports"

python -W ignore train.py \
        --gpu \
        --num-workers $num_workers \
        --backbone $backbone \
        --train-data-path $train_data_path \
        --val-data-path $val_data_path \
        --optimizer $optimizer \
        --num-epochs $num_epochs \
        --batch-size $batch_size \
        --lr $lr \
        --momentum $momentum \
        --weight-decay $weight_decay \
        --ckpt-save-path $ckpt_save_path \
        --ckpt-save-freq $ckpt_save_freq \
        --ckpt-prefix $ckpt_prefix \
        --report-path $report_path \
        --show-all-angles \
        --shuffle-data
        # --ckpt-load-path $ckpt_load_path

