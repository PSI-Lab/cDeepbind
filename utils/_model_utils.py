import numpy as np
import tensorflow as tf


def create_residual_block(input_node, l=16, w=16, gated=True):
    ar = np.random.choice([1, 2, 4])
    input_node = tf.keras.layers.BatchNormalization()(input_node)
    if gated:
        act_tanh = tf.keras.layers.Conv1D(l, w, dilation_rate=ar, padding='same', activation='tanh')(input_node)
        act_sigmoid = tf.keras.layers.Conv1D(l, w, dilation_rate=ar, padding='same', activation='sigmoid')(input_node)
        output = tf.keras.layers.multiply([act_tanh, act_sigmoid])
    else:
        output = tf.keras.layers.Conv1D(l, w, padding='SAME', dilation_rate=ar,
                                        activation='relu')(input_node)
    output = tf.keras.layers.add([output, input_node])
    return output


def create_model(name=None):
    W = 8
    L = 16
    input_layer = tf.keras.layers.Input(shape=(None, 4))
    input_node = tf.keras.layers.Conv1D(L, 1, padding='SAME')(input_layer)
    skip = tf.keras.layers.Conv1D(L, 1)(input_node)
    for i in range(W):
        input_node = create_residual_block(input_node)
        if (i + 1) % 4 == 0 or ((i + 1) == W):
            # Skip connections to the output after every 4 residual units
            dense = tf.keras.layers.Conv1D(L, 1)(input_node)
            skip = tf.keras.layers.add([skip, dense])
    skip = tf.keras.layers.Conv1D(L, 1, activation='relu')(skip)
    output = tf.keras.layers.GlobalMaxPool1D()(skip)
    output = tf.keras.layers.Dense(1, activation='linear')(output)
    model = tf.keras.models.Model(inputs=[input_layer],
                                  outputs=[output])
    optimizer = tf.keras.optimizers.Adam(1e-3)
    model.compile(optimizer, loss='mse')
    return model

