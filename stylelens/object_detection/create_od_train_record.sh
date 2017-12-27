#source activate bl-magi
export PYTHONPATH=$PYTHONPATH:`pwd`/../../tensorflow:`pwd`/../../slim
export DB_DATASET_HOST="35.187.193.199"
export DB_DATASET_NAME="bl-db-dataset"
export DB_DATASET_PORT="27017"
#export DATASET_PATH='gs://bluelens-style-model/object_detection'

echo $PYTHONPATH

python create_tf_record_from_db.py \
    --train_output_path=./train.record \
    --eval_output_path=./eval.record
