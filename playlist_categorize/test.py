
from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing.image import ImageDataGenerator
import os
import pathlib
from keras.models import load_model
from numpy import argmax

IMG_HEIGHT = 150
IMG_WIDTH = 150

def retrieve_predict(model_name, path, total_test):
    test_dir = os.path.join(path, 'test')
    model = load_model(os.path.join(path, 'model', model_name))
    test_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
    test_data_gen = test_image_generator.flow_from_directory(batch_size=1,
                                                             directory=os.path.join(test_dir),
                                                             shuffle=False,
                                                             target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                             class_mode='categorical')
    test_data_gen.reset()
    predict = model.predict_generator(test_data_gen, steps=total_test)
    return predict


def do_test(path, model_name, states, default_state):
    test_dir = os.path.join(path, 'test')
    all_test_files = sorted(list(pathlib.Path(test_dir).rglob('*.jpg')))
    total_test = len(all_test_files)
    predict = retrieve_predict(model_name, path, total_test)
    print_best_guess(all_test_files, default_state, predict, states)


def print_best_guess(all_test_files, default_state, predict, states):
    total_test = len(all_test_files)
    current_episode = 0
    for index in range(total_test):
        current_file = os.path.basename(all_test_files[index])

        episode, second_tot = map(int, (current_file.split('.')[0]).split('_')[1:3])
        if (episode != current_episode):
            current_episode = episode
            last_change = '00:00'
            prev_time = '00:00'
            current_state_map = {}
            skip = 0
            current_state = default_state
            print(f"================== {episode} ======================")

        hour, minute, second = second_tot // 3600, (second_tot // 60) % 60, second_tot % 60
        time_tpl = map(str, (hour, minute, second)) if hour > 0 else map(str, (minute, second))
        retrieved_state = argmax(predict[index])
        current_state_map[retrieved_state] = current_state_map.get(retrieved_state, 0) + 1

        current_time = ':'.join([x.zfill(2) for x in time_tpl])
        if retrieved_state != current_state:
            if prev_time != last_change and states[current_state] != states[default_state]:
                print('{}-{}'.format(last_change, prev_time), states[current_state])
                current_state_map = {}
                skip = 0
            current_state = retrieved_state
            last_change = current_time
        prev_time = current_time



