import tensorflow as tf
import os


def read_tfrecord(path):
  class_label1 = 0
  class_label2 = 0
  class_label3 = 0
  idx = 0
  for serialized_example in tf.python_io.tf_record_iterator(path):
    idx += 1
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    filename = example.features.feature["image/filename"].bytes_list.value
    label = example.features.feature["image/object/class/label"].int64_list.value
    
    if(label[0] == 1): 
      class_label1 += 1
    elif(label[0] == 2): 
      class_label2 += 1
    elif(label[0] == 3): 
      class_label3 += 1

    print("idx: {}, filename: {}, label: {}".format(idx, filename, label))
  
  print("label num")
  print("label 1: {}, label 2: {}, label 3: {}".format(class_label1, class_label2, class_label3))



current_path = os.getcwd()
train_input_file = os.path.join(current_path, "train.record")
test_input_file = os.path.join(current_path, "eval.record")

read_tfrecord(train_input_file)
#read_tfrecord(test_input_file)

