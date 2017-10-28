#! /bin/bash

source activate bl-magi
export PYTHONPATH=$PYTHONPATH:`pwd`/../../tensorflow:`pwd`/../../slim
#export DATASET_PATH='gs://bluelens-style-model/object_detection'
export DATASET_PATH='/dataset/deepfashion3'

python ./train.py \
    --logtostderr \
    --pipeline_config_path=$DATASET_PATH/data/ssd_inception_v2_3class.config \
    --train_dir=$DATASET_PATH/models/model/train
