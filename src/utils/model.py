import tensorflow as tf
import time
import os

def create_model(loss_function, optimizer, metrics):
    LAYERS = [tf.keras.layers.Flatten(input_shape = (28,28), name = "Input Layer"),
        tf.keras.layers.Dense(300, activation='relu', name = "Hidden Layer 1"),
        tf.keras.layers.Dense(100, activation='relu', name = "Hidden Layer 2"),
        tf.keras.layers.Dense(10, activation='softmax', name = "Output Layer"),
        ]


    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.summary()

    model_clf.compile(metrics = metrics, optimzier = optimizer, loss_function= loss_function)

    return model_clf

def unique_file_name(model_name):
    file_name = time.strftime(f"%Y%m%d_%H%M%S_{model_name}")
    return file_name

def save_model(model,model_name,model_dir):
    file_name = unique_file_name(model_name)
    model_dir_path = os.path.join(model_dir, file_name)
    model.save(model_dir_path)
    
