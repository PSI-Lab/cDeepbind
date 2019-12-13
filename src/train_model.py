import os
import argparse
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
from time import gmtime, strftime
from sklearn.model_selection import KFold


def create_residual_block(input_node, l, w, gated=True):
    ar = np.random.choice([1, 2, 4])
    input_node = tf.keras.layers.BatchNormalization()(input_node)
    if gated:
        act_tanh = tf.keras.layers.Conv1D(l, w, dilation_rate=ar, padding='same', activation='tanh')(input_node)
        act_sigmoid = tf.keras.layers.Conv1D(l, w, dilation_rate=ar, padding='same', activation='sigmoid')(input_node)
        output = tf.keras.layers.multiply([act_tanh, act_sigmoid])
    else:
        output = tf.keras.layers.Conv1D(l, w, padding='SAME', dilation_rate=ar,
                                        activation='relu')(input_node)
    output = squeeze_excite_block(output)
    output = tf.keras.layers.add([output, input_node])
    return output


def squeeze_excite_block(input_node, ratio=16):
    #     init = input_node
    filters = input_node.shape[-1]
    se_shape = (1, filters)

    se = tf.keras.layers.GlobalAvgPool1D()(input_node)
    se = tf.keras.layers.Reshape(se_shape)(se)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = tf.keras.layers.multiply([input_node, se])
    return x


def create_model(num_heads, input_dim, name=None):
    D = np.random.choice([6,12,18])
    W = np.random.choice([16,24,32])
    L = np.random.choice([32,64,128])
    gated= np.random.choice([True, False])

    input_layer = tf.keras.layers.Input(shape=(None, input_dim))
    input_node = tf.keras.layers.Conv1D(L, 1, padding='SAME')(input_layer)
    skip = tf.keras.layers.Conv1D(L, 1)(input_node)
    for i in range(D):
        input_node = create_residual_block(input_node, l=L, w=W, gated=gated)
        if (i + 1) % 3 == 0 or ((i + 1) == D):
            # Skip connections to the output after every 3 residual units
            dense = tf.keras.layers.Conv1D(L, 1)(input_node)
            skip = tf.keras.layers.add([skip, dense])
    skip = tf.keras.layers.Conv1D(L, 1, activation='relu')(skip)
    skip = tf.keras.layers.Conv1D(num_heads, 1, activation='relu')(skip)
    output_max = tf.keras.layers.GlobalMaxPool1D()(skip)
    output_avg = tf.keras.layers.GlobalAvgPool1D()(skip)
    output = tf.keras.layers.concatenate([output_max,output_avg])
    output = tf.keras.layers.Dense(num_heads, activation='linear')(output)
    model = tf.keras.models.Model(inputs=[input_layer],
                                  outputs=[output],
                                  name=name)
    optimizer = tf.keras.optimizers.Adam(1e-3)
    model.compile(optimizer, loss=MaskedLoss(num_heads, type='masked_huber_loss'))
    return model


class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self, num_heads, type='masked_mean_squared_error'):
        assert type in ['masked_mean_square_error',
                        'masked_huber_loss'], 'Invalid Loss Function'
        super(MaskedLoss, self).__init__(name=type)
        if type == 'masked_mean_square_error':
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        elif type == 'masked_huber_loss':
            self.loss_fn = tf.keras.losses.Huber()
        else:
            raise ValueError
        self.num_heads=num_heads

    def call(self, y_true, y_pred):
        #         drop_mask = tf.keras.layers.Dropout(0.9,noise_shape=(y_true.shape[0],1))(tf.ones_like(y_true))

        # drop_mask = np.random.choice([0, 1], size=(1, self.num_heads), p=[0.1, 0.9])
        drop_mask = np.ones(shape=(1, self.num_heads))
        y_true = tf.keras.layers.Flatten()(y_true * drop_mask)
        y_pred = tf.keras.layers.Flatten()(y_pred * drop_mask)

        mask = tf.math.logical_not(tf.math.is_nan(y_true))
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        return self.loss_fn(y_true=y_true, y_pred=y_pred)


def nanpearson(y_pred, y_true):
    mask = ~np.isnan(y_true)
    corr = []
    for col in range(mask.shape[1]):
        y_pred_col = y_pred[mask[:,col],col]
        y_true_col = y_true[mask[:,col], col]
        corr.append(pearsonr(y_pred_col, y_true_col)[0])
    return corr


def main():
    inf = np.load('data/rna_compete_2013_multiclass_False_position_only_False.npz')
    x = inf['x']
    y = inf['y']
    x_test = inf['x_test']
    y_test = inf['y_test']
    rbp_names = inf['rbp_names']
    num_heads = y.shape[1]
    input_dim = x.shape[-1]
    kf = KFold(n_splits=5)
    models = []
    best_loss = np.inf
    for train_ind, val_ind in kf.split():
        x_val = x[val_ind]
        y_val = y[val_ind]
        x_train = x[train_ind]
        y_train = y[train_ind]

        tictoc = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
        run_dir = 'model/debug/run_' + tictoc
        os.mkdir(run_dir)

        tf.keras.backend.clear_session()
        cnn_model = create_model(num_heads=num_heads,
                                 input_dim=input_dim)
        history = cnn_model.fit(x_train,y_train, validation_data=(x_val,y_val),
                    batch_size=128,
                    epochs=50,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3),
                                tf.keras.callbacks.CSVLogger('{}/history.csv'.format(run_dir)),
                                tf.keras.callbacks.History(),
                                tf.keras.callbacks.TensorBoard(log_dir=run_dir),
                                tf.keras.callbacks.ModelCheckpoint(run_dir+'/checkpoint.{epoch:03d}.hdf5',
                                save_best_only=False, period=1)]
                    )
        if history.history['val_loss'] < best_loss:
            best_model = cnn_model

    assert best_model, "This should have been assigned"
    y_pred = best_model.predict(x_test, batch_size=128)
    df_perf = pd.read_csv('data/cdeepbind_perf_comparison.csv')
    df_perf = df_perf.rename(columns={'RNAcompet experiment':'experiment'})
    df_perf = df_perf.set_index('experiment')
    df_perf['cDeepBind'] = np.nan
    df_perf.loc[rbp_names, 'cDeepBind'] = nanpearson(y_pred, y_test)
    df_perf.describe().to_csv('{}/perf_summary.csv'.format(run_dir))
    tf.keras.models.save_model(best_model,'{}/model.h5'.format(run_dir), include_optimizer=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_trials",
                        default=1,
                        help="Number of random hyperparameter trials to run",
                        type=int)
    args = parser.parse_args()
    for i in tqdm(range(args.num_trials)):
        main()
