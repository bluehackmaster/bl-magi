import os
from bson.objectid import ObjectId
from stylelens_dataset.images import Images
from util import s3
from PIL import Image

CATEGORY_CLOTH_FILE = './list_category_cloth.txt'
CATEGORY_IMG_FILE = './list_category_img.txt'
ATTR_CLOTH_FILE = './list_attr_cloth.txt'
ATTR_IMG_FILE = './list_attr_img.txt'
BBOX_FILE = './list_bbox.txt'

AWS_BUCKET = 'stylelens-dataset'

AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY'].replace('"', '')
AWS_SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY'].replace('"', '')
storage = s3.S3(AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY)

def get_category_clothes():
  category_cloth = open(CATEGORY_CLOTH_FILE, 'r')
  category_clothes = []
  for pair in category_cloth.readlines():
    map = pair.strip().split()
    category = {}
    category['name'] = map[0]
    category['type'] = map[1]
    category_clothes.append(category)
  return category_clothes

def get_category_images(category_clothes):
  category_img = open(CATEGORY_IMG_FILE, 'r')
  category_images = []
  for pair in category_img.readlines():
    cc = category_clothes
    map = pair.strip().split()
    category = {}
    category['file'] = map[0]
    category['name'] = cc[int(map[1]) - 1]['name']
    category['type'] = cc[int(map[1]) - 1]['type']
    category_images.append(category)
  return category_images

def get_attribute_clothes():
  attr_cloth = open(ATTR_CLOTH_FILE, 'r')
  attribute_clothes = []
  for pair in attr_cloth.readlines():
    map = pair.strip().rsplit(' ', 1)
    attr = {}
    attr['name'] = map[0].strip()
    attr['type'] = map[1]
    attribute_clothes.append(attr)
  return attribute_clothes

def get_attribute_images(attribute_clothes):
  print('get_attribute_images:start')
  attr_img = open(ATTR_IMG_FILE, 'r')
  attribute_images = []
  for pair in attr_img.readlines():
    map = pair.strip().split()
    attr = {}
    attr['texture_class'] = []
    attr['fabric_class'] = []
    attr['shape_class'] = []
    attr['part_class'] = []
    attr['style_class'] = []

    attr['file'] = map[0]
    for i in range(1, 1001):
      if map[i] == '1':
        ac = attribute_clothes[i - 1]
        if ac['type'] == '1':
          attr['texture_class'].append(ac['name'])
        elif ac['type'] == '2':
          attr['fabric_class'].append(ac['name'])
        elif ac['type'] == '3':
          attr['shape_class'].append(ac['name'])
        elif ac['type'] == '4':
          attr['part_class'].append(ac['name'])
        elif ac['type'] == '5':
          attr['style_class'].append(ac['name'])

    attribute_images.append(attr)

  return attribute_images

def get_bbox():
  bbox = open(BBOX_FILE, 'r')
  bboxes = []
  for pair in bbox.readlines():
    map = pair.strip().split()
    box = {}
    box['file'] = map[0].strip()
    box['x1'] = int(map[1])
    box['y1'] = int(map[2])
    box['x2'] = int(map[3])
    box['y2'] = int(map[4])
    bboxes.append(box)
  return bboxes

def upload_image_to_storage(id, img_file):
  print("upload_image_to_storage")
  print("id = " + str(id))
  print("img_file = " + img_file)
  file = '/dataset/' + img_file
  key = os.path.join('deepfashion', 'img', id + '.jpg')
  is_public = True
  file_url = storage.upload_file_to_bucket(AWS_BUCKET, file, key, is_public=is_public)
  return file_url

def get_image_size(img_file):
  with Image.open(img_file) as img:
    return img.size

if __name__ == '__main__':
  print('start')

  dataset_api = Images()
  bboxes = get_bbox()

  category_clothes = get_category_clothes()
  category_images = get_category_images(category_clothes)

  attribute_clothes = get_attribute_clothes()
  attribute_images = get_attribute_images(attribute_clothes)

  i = 0
  for img in attribute_images:
    image = {}
    image['file'] = img['file']
    image['source'] = 'deepfashion'
    width, height = get_image_size(img['file'])
    image['width'] = width
    image['height'] = height
    image['category_class'] = category_images[i]['type']
    image['category_name'] = category_images[i]['name']
    image['texture_class'] = attribute_images[i]['texture_class']
    image['fabric_class'] = attribute_images[i]['fabric_class']
    image['shape_class'] = attribute_images[i]['shape_class']
    image['part_class'] = attribute_images[i]['part_class']
    image['style_class'] = attribute_images[i]['style_class']
    image['bbox'] = bboxes[i]
    i = i + 1

    try:
      id = dataset_api.add_image(image)
      url = upload_image_to_storage(id, image['file'])

      image['_id'] = ObjectId(id)
      image['url'] = url
      dataset_api.update_image(image)
    except Exception as e:
      print(e)
