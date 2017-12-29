#! /bin/bash

source activate bl-magi
#export PYTHONPATH=$PYTHONPATH:`pwd`/../../tensorflow:`pwd`/../../slim
export PYTHONPATH=$PYTHONPATH:`pwd`/../../tensorflow:`pwd`/../../tensorflow/slim:'pwd'/../../tensorflow/object_detection
#export MODEL_BASE_PATH='gs://bluelens-style-model/object_detection'
export MODEL_BASE_PATH='/home/lion/dataset/deepfashion3'

python ./train.py \
    --logtostderr \
    --pipeline_config_path=$MODEL_BASE_PATH/models/model/ssd_inception_v2_3class.config \
    --train_dir=$MODEL_BASE_PATH/models/model/train
