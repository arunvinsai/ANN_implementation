from src.utils.common import read_yaml_contents
import argparse

def training(config_path):
    config = read_yaml_contents(config_path)
    print(config)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()
    training(config_path = parsed_args.config)
