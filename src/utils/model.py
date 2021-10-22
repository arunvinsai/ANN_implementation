import tensorflow as tf

def create_model(loss_function, optimizer, metrics):
    LAYERS = [tf.keras.layers.Flatten(input_shape = input_shape, name = "Input Layer"),
        tf.keras.layers.Dense(300, activation='relu', name = "Hidden Layer 1"),
        tf.keras.layers.Dense(100, activation='relu', name = "Hidden Layer 2"),
        tf.keras.layers.Dense(10, activation='softmax', name = "Output Layer"),
        ]


    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.summary()

    model_clf.compile(metrics = metrics, optimzier = optimizer, loss_function= loss_function)

    return model_clf
