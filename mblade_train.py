
from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing.image import ImageDataGenerator
import os
import yaml
import pathlib

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import glob

num_classes=6


with open('config.yml') as f:
  config = yaml.safe_load(f)

PATH = config['PATH']
PATH_LIB = pathlib.Path(PATH)

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

total_train = len(list(pathlib.Path(train_dir).rglob('*.jpg')))
total_val = len(list(pathlib.Path(validation_dir).rglob('*.jpg')))

print(total_train, total_val)

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data


train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')



val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

print(train_data_gen.class_indices, train_data_gen.classes)
print(val_data_gen.class_indices, val_data_gen.classes)



model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)


model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

os.makedirs(os.path.join(PATH, 'model'), exist_ok=True)
model.save(os.path.join(PATH, 'model', 'mblade_categorizer.hdf5'))