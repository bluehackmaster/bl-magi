#! /bin/bash

source activate bl-magi
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python ./object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=./ssd_inception_v2_ppl.config \
    --train_dir=/dataset/models/model/train
