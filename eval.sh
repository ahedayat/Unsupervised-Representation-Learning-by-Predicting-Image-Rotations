
# Hardware
num_workers=2

# Architecture
backbone="resnet34"

# Data Path
train_data_path="/content/datasets/cifar10"
val_data_path="/content/datasets/cifar10"

# Model Parameters
feature_layer_index=2

# Optimizer Parameters
optimizer="adam"
num_epochs=100
batch_size=1024

# Learning Rate 
lr=1e-3
momentum=0.9
weight_decay=5e-4

# Checkpoints Parameters
ckpt_save_path="./checkpoints_classification"
ckpt_load_path="./checkpoints_all/rotation_final_all.ckpt"
ckpt_prefix="classification"
ckpt_save_freq=20

# Report
report_path="./reports_classification_all_2"

python -W ignore eval.py \
        --gpu \
        --num-workers $num_workers \
        --backbone $backbone \
        --train-data-path $train_data_path \
        --val-data-path $val_data_path \
        --feature-layer-index $feature_layer_index \
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
        --ckpt-load-path $ckpt_load_path \
        --show-all-angles 
        # --shuffle-data 

