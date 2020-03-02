
from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing.image import ImageDataGenerator
import os
import pathlib
from keras.models import load_model
from numpy import argmax

IMG_HEIGHT = 150
IMG_WIDTH = 150
SIMPLE_STATES = {0: 'FIGHTS', 1: 'PEACE'}

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

def process_for_suggestion_list(episode, default_state, states):
    second_tot = 2
    current_state = default_state
    last_change = '00:02'
    prev_time = '00:02'
    skip = 0
    for retrieved_state in episode:
        hour, minute, second = second_tot // 3600, (second_tot // 60) % 60, second_tot % 60
        time_tpl = map(str, (hour, minute, second)) if hour > 0 else map(str, (minute, second))
        current_time = ':'.join([x.zfill(2) for x in time_tpl])
        if retrieved_state != current_state:
            if prev_time != last_change and current_state != default_state:
                print('{}-{}'.format(last_change, prev_time), states[current_state])
            current_state = retrieved_state
            last_change = current_time
        prev_time = current_time
        second_tot += 2

def do_test(path, model_name, states, default_state):
    test_dir = os.path.join(path, 'test')
    all_test_files = sorted(list(pathlib.Path(test_dir).rglob('*.jpg')))
    total_test = len(all_test_files)
    predict = retrieve_predict(model_name, path, total_test)
    predict_fight = argmax(predict, axis=1)
    episodes = get_state_list(all_test_files, predict)
    for episode_key in episodes:
        print(f"================= {episode_key} ========================")
        print(episodes[episode_key])
        simple_episode = [1 if state == default_state else 0 for state in episodes[episode_key]]
        print(simple_episode)

        process_for_suggestion_list(simple_episode, 1, SIMPLE_STATES)
        print(f"================= {episode_key} ========================")
        process_for_suggestion_list(episodes[episode_key], default_state, states)


def get_state_list(all_test_files,  predict):
    total_test = len(all_test_files)
    current_episode = 0
    episodes = {}
    for index in range(total_test):
        current_file = os.path.basename(all_test_files[index])
        episode, second_tot = map(int, (current_file.split('.')[0]).split('_')[1:3])
        if (episode != current_episode):
            current_episode = episode
            episodes[episode] = []
        retrieved_state = argmax(predict[index])
        episodes[episode].append(retrieved_state)
    return episodes


