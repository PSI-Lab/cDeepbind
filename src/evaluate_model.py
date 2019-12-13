import os
import argparse
from tqdm import tqdm
import datacorral as dc
import pandas as pd
import tensorflow as tf
import numpy as np
from scipy.stats import pearsonr
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def nanpearson(y_pred, y_true):
    mask = ~np.isnan(y_true)
    corr = []
    for col in range(mask.shape[1]):
        y_pred_col = y_pred[mask[:, col], col]
        y_true_col = y_true[mask[:, col], col]
        corr.append(pearsonr(y_pred_col, y_true_col)[0])
    return corr


def plot(df_perf):
    f, ax = plt.subplots(figsize=(18, 12))
    predictors = ['DeepBind',
                               'RNAcontext_org',
                               'RCK',
                               'DLPRB-CNN',
                               'DLPRB-RNN',
                               'cDeepBind',
                               ]
    df_merged = pd.concat([pd.DataFrame(data={'predictor': p, 'correlation': df_perf[p].values}) for p in predictors],
                          sort=False)
    sns.violinplot(x="predictor", y="correlation", data=df_merged, inner="box", palette="Set3", cut=2, linewidth=3, )
    f.suptitle('Model Performance on held out probes', fontsize=18, fontweight='bold')
    ax.set_xlabel("Predictor", size=16, alpha=0.7)
    ax.set_ylabel("Correlation", size=16, alpha=0.7)
    plt.savefig('plots/model_comparison.png', dpi=200)


def main():
    # df = pd.read_csv(dc.Client().get_path('3gsL9R'), compression='gzip')
    # rbp_names = [col for col in df.columns if 'RNCMPT' in col]
    inf = np.load('data/rna_compete_2013/encoded_data_new_struct.npz')
    x_test = inf['x_test']
    y_test = inf['y_test']
    rbp_names = inf['rbp_names']
    model_paths = ['model/positional_structure/run_2019_10_03_23_56_39/model.h5',
                   'model/positional_structure/run_2019_10_03_21_23_39/model.h5',
                   'model/positional_structure/run_2019_10_03_20_58_26/model.h5',
                   'model/positional_structure/run_2019_10_03_23_05_51/model.h5',
                   'model/positional_structure/run_2019_10_03_22_25_33/model.h5']

    trained_models = []
    for path in tqdm(model_paths):
        trained_models.append(tf.keras.models.load_model(path, compile=False))
    y_pred = np.mean([model.predict(x_test, batch_size=256, verbose=1) for model in trained_models], axis=0)
    df_perf = pd.read_csv('data/cdeepbind_perf_comparison.csv')
    df_perf = df_perf.rename(columns={'RNAcompet experiment': 'experiment'})
    df_perf = df_perf.set_index('experiment')
    df_perf['cDeepBind-new'] = np.nan
    df_perf.loc[rbp_names, 'cDeepBind-new'] = nanpearson(y_pred, y_test)
    plot(df_perf)
    df_perf.to_csv('data/cdeepbind_new_perf_comparison.csv',
                   index_label='experiment')


if __name__ == '__main__':
    main()
