import yaml
from playlist_categorize.train import do_train
from playlist_categorize.test import do_test
import argparse
import sys


with open('config.yml') as f:
  config = yaml.safe_load(f)
  path = config['PATH']
  num_classes = config['NUM_CLASSES']
  model_name = config['MODEL_NAME']
  states = config['STATES']
  default_state = 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('command', help='Subcommand to run')
    args = parser.parse_args(sys.argv[1:2])
    if args.command not in ['do_train', 'do_test', 'do_all']:
        print('Unrecognized command')
        parser.print_help()
        exit(1)

    if args.command in ['do_train', 'do_all']:
        do_train(path, num_classes, model_name)
    if args.command in ['do_test', 'do_all']:
        do_test(path, model_name, states, default_state)
