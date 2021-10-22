import tensorflow as tf

def get_data(validataion_data_size):
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.laod_data()

    #Data splitting for cross validataion and training and Normalising the data points
    X_train_val = X_train_full[:validataion_data_size]/255.
    y_train_val = y_train_full[:validataion_data_size]/255.

    X_train = X_train_full[validataion_data_size:]/255.
    y_train = y_train_full[validataion_data_size:]/255.

    X_test = X_test/255.

    return (X_train, y_train), (X_train_val, y_train_val), (X_test, y_test)
