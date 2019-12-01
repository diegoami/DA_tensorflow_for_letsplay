
from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing.image import ImageDataGenerator
import os
import pathlib
from keras.models import load_model
from numpy import argmax


def do_test(path, model_name, states, default_state):
    test_dir = os.path.join(path, 'test')
    all_test_files = sorted(list(pathlib.Path(test_dir).rglob('*.jpg')))
    total_test = len(all_test_files)
    IMG_HEIGHT = 150
    IMG_WIDTH = 150
    model = load_model(os.path.join(path, 'model', model_name))
    test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    test_data_gen = test_image_generator.flow_from_directory(batch_size=1,
                                                             directory=os.path.join(test_dir),
                                                             shuffle=False,
                                                             target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                             class_mode='categorical')
    test_data_gen.reset()
    predict = model.predict_generator(test_data_gen, steps=total_test)
    current_state = default_state
    last_change = '00:00'
    prev_time = '00:00'
    for index in range(total_test):
        current_file = os.path.basename(all_test_files[index])
        episode, second_tot = map(int, (current_file.split('.')[0]).split('_')[1:3])
        hour, minute, second = second_tot // 3600, (second_tot // 60) % 60, second_tot % 60
        time_tpl = map(str, (hour, minute, second)) if hour > 0 else map(str, (minute, second))
        # total_predict = predict[index]
        # if index > 0:
        #    total_predict += predict[index-1]*0.5
        # if index < total_test-1:
        #    total_predict += predict[index+1]*0.5
        new_state = argmax(predict[index])
        current_time = ':'.join([x.zfill(2) for x in time_tpl])
        if new_state != current_state:
            if prev_time != last_change and states[current_state] != states[default_state]:
                print('{}-{}'.format(last_change, prev_time), states[current_state])
            current_state = new_state
            last_change = current_time
        prev_time = current_time