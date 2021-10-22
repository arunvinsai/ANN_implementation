import yaml

def read_yaml_contents(config_path):
    with open(config_path) as fp:
        contents = yaml.safe_load(fp)
    return contents
