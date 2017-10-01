#! /bin/bash

source activate bl-magi
export PYTHONPATH=$PYTHONPATH:`pwd`/../../tensorflow:`pwd`/../../slim
export MODEL_BASE_PATH='gs://bluelens-style-model/object_detection'

python ./eval.py \
    --logtostderr \
    --pipeline_config_path=$MODEL_BASE_PATH/data/faster_rcnn_resnet152.config \
    --checkpoint_dir=$MODEL_BASE_PATH/models/model/train \
    --eval_dir=$MODEL_BASE_PATH/models/model/eval

