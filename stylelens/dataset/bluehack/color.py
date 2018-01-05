import os
import uuid
import urllib.request
from PIL import Image
from util import s3

from stylelens_dataset.colors import Colors

AWS_DATASET_BUCKET = 'stylelens-dataset'
AWS_DIR = 'bluehack/color'

IMG_WIDTH = 300
IMG_HEIGHT = 300

AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY'].replace('"', '')
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY'].replace('"', '')

storage = s3.S3(AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY)

# ex) stylelens-dataset / bluehack / color / red / xxxxx.jpg
def save_image_to_storage(color_name, file_name):
  file = file_name + '.jpg'
  key = os.path.join(AWS_DIR, color_name, file_name + '.jpg')
  is_public = True
  file_url = storage.upload_file_to_bucket(AWS_DATASET_BUCKET, file, key, is_public=is_public)
  return file_url


def crawl_from_google_images()

  data = []

  item = {}
  item['url'] = 'http://www.google.com/xxx/yyy/zzz.jpg'
  item['color_name'] = 'red'
  item['color_code'] = 'CC0200'
  data.append(item)

  return data

def download_image(url):
  try:
    # f = urllib.urlopen(urllib.quote(image_path.encode('utf8'), '/:'))
    f = urllib.request.urlopen(url)
    im = Image.open(f).convert('RGB')
    im = im.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
    file_name = str(uuid.uuid4()) + '.jpg'
    im.save(file_name)
  except Exception as e:
    print(e)

  return file_name

if __name__ == '__main__':
  try:
    api_instance = Colors()

    data = crawl_from_google_images()

    for item in data:
      file_name = download_image(item['url'])
      file = save_image_to_storage(item['color_name'], file_name)

      color = {}
      color['file'] = file
      color['name'] = item['color_name']
      color['code'] = item['color_code']
      api_response = api_instance.add_color(color)
  except Exception as e:
    print("Exception when calling add_color: %s\n" % e)

