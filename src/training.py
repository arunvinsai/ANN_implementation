from src.utils.common import read_yaml_contents
from src.utils.load_data import get_data
from src.utils.model import create_model, save_model
import argparse
import os

def training(config_path):
    config = read_yaml_contents(config_path)
    print(config)

    validation_datasize = config['params']['validation_data_size']

    (X_train, y_train), (X_train_val, y_train_val), (X_test, y_test) = get_data(validation_datasize)

    epochs = config['params']['epochs']
    loss_function = config['params']['loss_function']
    metrics = config["params"]["metrics"]
    optimizer = config['params']['optimizer']

    model = create_model(loss_function, metrics, optimizer)
    history = model.fit(X_train, y_train, epochs = epochs, validation_data= (X_train_val, y_train_val) )
    
    artifacts_dir = config['artifacts']['artifacts_dir']
    model_dir = config['artifacts']['model_dir']
    model_name = config['artifacts']['model_name']

    model_dir_path = os.path.join(artifacts_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    # saving the model
    save_model(model, model_name, model_dir_path)




if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()
    training(config_path = parsed_args.config)
