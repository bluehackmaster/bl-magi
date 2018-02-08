import argparse
import os

import PIL.Image as Image
import numpy as np
import tensorflow as tf
from utils.app_utils import FPS

from dataset.df import visualization_utils as vis_util
from object_detection.utils import label_map_util

CWD_PATH = os.getcwd()
detection_graph = tf.Graph()

IMAGE_SIZE = (12, 8)
# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_CKPT = os.path.join(CWD_PATH, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'label_map.pbtxt')

NUM_CLASSES = 89

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def detect_objects(image_np, sess):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2)
    return image_np


def start_tf():
  # detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

  return sess

def stop_tf(sess):
  sess.close()


def inference(sess, img_np):
    fps = FPS().start()
    fps.update()
    output = detect_objects(img_np, sess)
    display_PIL(output)
    fps.stop()


def display_PIL(nparray):
  image = Image.fromarray(nparray)
  image.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    sess = start_tf()
    for file in os.listdir("/Users/bok95/testimages"):
      if file.endswith(".jpg"):
        image = Image.open(os.path.join('/Users/bok95/testimages', file))
        img_np = load_image_into_numpy_array(image)
        inference(sess, img_np)

    stop_tf(sess)
