
from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing.image import ImageDataGenerator
import os
import yaml
import pathlib
from keras.models import load_model
from numpy import argmax

num_classes=3


with open('config.yml') as f:
  config = yaml.safe_load(f)

PATH = config['PATH']
PATH_LIB = pathlib.Path(PATH)

test_dir = os.path.join(PATH, 'test')
validation_dir = os.path.join(PATH, 'validation')

all_test_files = sorted(list(pathlib.Path(test_dir).rglob('*.jpg')))
total_test = len(all_test_files)
STATES = ['BATTLE', 'HIDEOUT', 'UNKNOWN']

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

model = load_model(os.path.join(PATH, 'model', 'mblade_categorizer.hdf5'))
test_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data


test_data_gen = test_image_generator.flow_from_directory(batch_size=1,
                                                           directory=os.path.join(test_dir),
                                                           shuffle=False,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

test_data_gen.reset()
predict = model.predict_generator(test_data_gen, steps=total_test)
current_state = 2
last_change = '00:00'
prev_time = '00:00'
for index in range(total_test):
    current_file = os.path.basename(all_test_files[index])
    episode, second_tot = map(int, (current_file.split('.')[0]).split('_')[1:3])
    hour, minute, second = second_tot // 3600,  (second_tot // 60) % 60, second_tot % 60
    time_tpl = map(str, (hour, minute, second)) if hour > 0 else map(str,(minute, second))

    new_state = argmax(predict[index])
    current_time = ':'.join([x.zfill(2) for x in time_tpl])
    if new_state != current_state:
        if STATES[current_state] != 'UNKNOWN':
            print('{}-{}'.format(last_change, prev_time), STATES[current_state])
        current_state = new_state
        last_change = current_time
    prev_time = current_time