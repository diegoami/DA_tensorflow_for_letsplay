
from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing.image import ImageDataGenerator
import os
import yaml
import pathlib
from keras.models import load_model
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import glob

num_classes=3


with open('config.yml') as f:
  config = yaml.safe_load(f)

PATH = config['PATH']
PATH_LIB = pathlib.Path(PATH)

test_dir = os.path.join(PATH, 'test')
validation_dir = os.path.join(PATH, 'validation')

total_test = len(list(pathlib.Path(test_dir).rglob('*.jpg')))

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
