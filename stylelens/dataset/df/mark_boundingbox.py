import os
import urllib.request

import numpy as np
from PIL import Image
from stylelens_dataset.images import Images
from stylelens_s3.s3 import S3

from dataset.df import visualization_utils as vis_util

AWS_BUCKET = 'stylelens-dataset'

AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY'].replace('"', '')
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY'].replace('"', '')
storage = S3(AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY)

AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY'].replace('"', '')
# DB_DATASET_HOST = os.environ['DB_DATASET_HOST']
# DB_DATASET_NAME = os.environ['DB_DATASET_NAME']
# DB_DATASET_USER = os.environ['DB_DATASET_USER']
# DB_DATASET_PORT = os.environ['DB_DATASET_PORT']
# DB_DATASET_PASSWORD = os.environ['DB_DATASET_PASSWORD']


def download_image(path):
  try:
    f = urllib.request.urlopen(path)
  except Exception as e:
    print('download_image: ' + str(e))
    return None
  return f

def upload_image_to_storage(id, file):
  print("upload_image_to_storage")
  print("id = " + str(id))
  print("img_file = " + file)
  key = os.path.join('deepfashion', 'img', 'with_box', id + '.jpg')
  is_public = True
  file_url = storage.upload_file_to_bucket(AWS_BUCKET, file, key, is_public=is_public)
  return file_url

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
    (im_height, im_width, 3)).astype(np.uint8)


if __name__ == '__main__':
  print('start')
  image_api = Images()
  offset = 0
  limit = 100

  while True:
    images = None
    try:
      images = image_api.get_images_by_source("deepfashion", offset, limit)
    except Exception as e:
      print(str(e))

    if len(images) == 0:
      break
    else:
      offset = offset + limit

    for img in images:
      img_file = download_image(img.get('url'))
      image = Image.open(img_file)
      image_np = load_image_into_numpy_array(image)
      bbox = img.get('bbox')
      boxes = np.array([[bbox.get('y1'), bbox.get('x1'), bbox.get('y2'), bbox.get('x2')]])

      vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        boxes,
        use_normalized_coordinates=False,
        line_thickness=5)

      image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
      # image_pil.show()
      TMP_FILE = 'tmp.jpg'
      image_pil.save('tmp.jpg')
      file_with_box = upload_image_to_storage(str(img.get('_id')), TMP_FILE)
      os.remove(TMP_FILE)

      image_data = {}
      image_data['_id'] = img.get('_id')
      image_data['url_with_box'] = file_with_box
      try:
        image_api.update_image(image_data)
      except Exception as e:
        print(str(e))




  # dataset_api.update_image(image)
