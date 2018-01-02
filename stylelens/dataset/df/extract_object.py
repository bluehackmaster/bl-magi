import os
from bson.objectid import ObjectId
from stylelens_dataset.images import Images
from stylelens_dataset.objects import Objects
from util import s3
import urllib.request
from PIL import Image

AWS_BUCKET = 'stylelens-dataset'

AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY'].replace('"', '')
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY'].replace('"', '')
storage = s3.S3(AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY)

TMP_FILE = 'tmp_file.jpg'

def upload_image_to_storage(file):
  print("upload_image_to_storage")
  key = os.path.join('deepfashion', 'obj', file)
  is_public = True
  file_url = storage.upload_file_to_bucket(AWS_BUCKET, file, key, is_public=is_public)
  return file_url

def get_image_size(img_file):
  with Image.open(img_file) as img:
    return img.size

def crop(image):
  file = image['url']

  bbox = image['bbox']
  left = bbox['x1']
  top = bbox['y1']
  right = bbox['x2']
  bottom = bbox['y2']

  try:
    f = urllib.request.urlopen(file)
    im = Image.open(f)
    area = (left, top, left + abs(left-right), top + abs(bottom-top))
    cropped = im.crop(area)
    cropped.save(TMP_FILE)
    uploaded_path = upload_image_to_storage(TMP_FILE)
    os.remove(TMP_FILE)
  except Exception as e:
    print(e)

  return uploaded_path


if __name__ == '__main__':
  print('start')

  image_api = Images()
  object_api = Objects()

  offset = 0
  limit = 100

  while True:
    try:
      res = image_api.get_images_by_source(source='deepfashion', offset=offset, limit=limit)

      for image in res:
        cropped_file = crop(image)
        image['url'] = cropped_file
        image['image_id'] = str(image['_id'])
        image['bbox'] = None
        image['width'] = None
        image['height'] = None
        object_api.add_object(image)

      if limit > len(res):
        break
      else:
        offset = offset + limit

    except Exception as e:
      print(e)
