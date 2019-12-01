import yaml
from playlist_categorize.train import do_train
from playlist_categorize.test import do_test




with open('config.yml') as f:
  config = yaml.safe_load(f)
  path = config['PATH']
  num_classes = config['NUM_CLASSES']
  model_name = config['MODEL_NAME']

#do_train(path, num_classes, model_name)
do_test(path, model_name)