from stylelens_dataset.cateories import Categories

TEXT_DATASET_FILE = './list_category_cloth.txt'

category_api = Categories()

def add_category(name, clazz):
  category = {}
  text['class_code'] = class_code
  text['text'] = keyword

  try:
    res = text_api.add_text(text)
  except Exception as e:
    print(e)

if __name__ == '__main__':
  try:
    text_dataset = open(TEXT_DATASET_FILE, 'r')
    texts = []
    for pair in text_dataset.readlines():
      map = pair.strip().split(' ', 1)
      tmp = map[1].strip().split(',')
      keywords = list(set(tmp))
      class_code = str(map[0])

      for keyword in keywords:
        print('' + class_code + ":" + keyword)
        add_text(class_code, keyword)

  except Exception as e:
    print(e)
