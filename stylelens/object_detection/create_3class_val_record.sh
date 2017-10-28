#!/bin/bash

source activate bl-magi
export PYTHONPATH=$PYTHONPATH:`pwd`/../../tensorflow:`pwd`/../../slim
#export MODEL_DATASET_PATH='gs://bluelens-style-model/object_detection'
export DATASET_PATH='/dataset/deepfashion3'

python create_tf_record.py \
    --label_map_path=$DATASET_PATH/data/label_map.pbtxt \
    --data_dir=$DATASET_PATH --folder=dataset --set=val \
    --output_path=$DATASET_PATH/data/val.record
