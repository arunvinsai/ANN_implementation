from src.utils.common import read_yaml_contents
from src.utils.load_data import get_data
import argparse

def training(config_path):
    config = read_yaml_contents(config_path)
    print(config)

    validation_datasize = config['params']['validatation_data_size']

    (X_train, y_train), (X_train_val, y_train_val), (X_test, y_test) = get_data(validation_datasize)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()
    training(config_path = parsed_args.config)
