r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    ./create_pascal_tf_record --data_dir=/home/user/VOCdevkit \
        --folder=merged \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
# import logging
import os
import random
# import re

import PIL.Image
import tensorflow as tf

from os import listdir
# from os.path import isdir, isfile, join

from object_detection.utils import dataset_util
# from object_detection.utils import label_map_util

from stylelens_dataset.images import Images
# from stylelens_dataset.objects import Objects


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('folder', 'merged', 'Desired challenge folder.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# flags.DEFINE_string('eval_output_path', '', 'Path to eval output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS
SETS = ['train', 'val', 'trainval', 'test']


def dict_to_tf_example(data, image_subdirectory='JPEGImages'):
  """
    Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
            running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
                                dataset  (default: False).
    1image_subdirectory: String specifying subdirectory within the
                        PASCAL dataset directory holding the actual image data.

    Returns:
        example: The converted tf.Example.

    Raises:
        ValueError: if the image pointed to by data['filename']
                    is not a valid JPEG
  """

  full_path = os.path.join('/dataset', data['file'])
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  poses = []
  classes = []
  classes_text = []

  width = int(data['width'])
  height = int(data['height'])
  xmin.append(float(data['bbox']['x1']) / width)
  ymin.append(float(data['bbox']['y1']) / height)
  xmax.append(float(data['bbox']['x2']) / width)
  ymax.append(float(data['bbox']['y2']) / height)
  classes_text.append(data['category_name'].encode('utf8'))
  classes.append(int(data['category_class']))
  difficult = int(0)
  truncated = int(0)
  poses.append('Frontal'.encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['file'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          str(data['_id']).encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  dataset_api = Images()
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  offset = 0
  limit = 50

  while True:
    try:
      res = dataset_api.get_images_by_source("deepfashion", offset=offset,
                                             limit=limit)
      for image in res:
        tf_data = dict_to_tf_example(image)
        writer.write(tf_data.SerializeToString())

      if limit > len(res):
        print("done")
        break
      else:
        offset = offset + len(res)
    except Exception as e:
      print(str(e))

  writer.close()


if __name__ == '__main__':
  tf.app.run()
