#!/bin/bash

source activate bl-magi
export PYTHONPATH=$PYTHONPATH:`pwd`/../../tensorflow:`pwd`/../../slim
export MODEL_BASE_PATH='gs://bluelens-style-model/object_detection'

python create_tf_record.py \
    --label_map_path=/dataset/deepfashion/data/label_map.pbtxt \
    --data_dir=/dataset/deepfashion --folder=dataset --set=train \
    --output_path=$MODEL_BASE_PATH/data/train.record
