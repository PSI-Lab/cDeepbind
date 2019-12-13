import argparse
import numpy as np
import datacorral as dc
import pandas as pd
from tqdm import tqdm
import RNA
import logging
import forgi

# create model details
md = RNA.md()
# activate unique multibranch loop decomposition
md.uniq_ML = 1
logging.basicConfig(level=logging.WARN)


def encode_sequence(s, method='standard', order='ACGT'):
    y = s.upper()
    # Signed encoding is the one described at
    # https://media.nature.com/original/nature-assets/nbt/journal/v33/n8/extref/nbt.3300-S2.pdf
    # Standard is the usual one-hot encoding
    assert method in ['standard', 'signed']
    assert all([x in 'ACGT' for x in order])
    assert set(list(order)) == set(list('ACGT'))

    ENCODING = np.zeros((5, 4))
    ENCODING[1:, :] = np.eye(4)
    if method == 'signed':
        ENCODING[1:, :] = ENCODING[1:, :] - 0.25
    _map = {x: i + 1 for i, x in enumerate(order)}
    _map['N'] = 0
    _map['U'] = _map['T']
    assert all([x in 'ACGTUN' for x in y])
    for x, i in _map.iteritems():
        y = y.replace(x, str(i))
    y = np.asarray(map(int, list(y)))
    return ENCODING[y.astype('int8')]


def one_hot_encode_struct(seq, order='PHIME', method='standard'):
    y = seq.upper()

    assert all([x in 'PHIME' for x in order])
    assert set(list(order)) == set(list('PHIME'))
    # Signed encoding is the one described at
    # https://media.nature.com/original/nature-assets/nbt/journal/v33/n8/extref/nbt.3300-S2.pdf
    # Standard is the usual one-hot encoding
    assert method in ['standard', 'signed']
    ENCODING = np.eye(5)
    if method == 'signed':
        ENCODING = ENCODING - 0.20

    _map = {x: i + 1 for i, x in enumerate(order)}
    assert all([x in 'PHIME' for x in y])
    for x, i in _map.iteritems():
        y = y.replace(x, str(i))
    y = np.asarray(map(int, list(y))) - 1
    return ENCODING[y.astype('int8')]


def create_multiclass_vector(seq, num_backtrack=10):
    # create fold compound object
    fc = RNA.fold_compound(seq, md)
    # compute MFE
    (ss, mfe) = fc.mfe()
    # rescale Boltzmann factors according to MFE
    fc.exp_params_rescale(mfe)
    # compute partition function to fill DP matrices
    fc.pf()
    structures = []
    dict_map = {'s': 'P',
                'i': 'I',
                'h': 'H',
                'm': 'M',
                'f': 'E',
                't': 'E'}
    one_hot_vector = np.zeros((len(seq), 5))
    for s in fc.pbacktrack(num_backtrack):
        structures.append(s)
        bg, = forgi.load_rna(s)
        element_string = bg.to_element_string()
        element_string = ''.join(map(lambda x: dict_map[x], element_string))
        one_hot_vector += one_hot_encode_struct(element_string, method='signed')
    one_hot_vector /= 1.0 * num_backtrack
    return one_hot_vector


def create_pairing_matrix(seq, num_backtrack =10):
    # create fold compound object
    fc = RNA.fold_compound(seq, md)
    # compute MFE
    (ss, mfe) = fc.mfe()
    # rescale Boltzmann factors according to MFE
    fc.exp_params_rescale(mfe)
    # compute partition function to fill DP matrices
    fc.pf()
    structures = []
    pair_tables = []
    pair_matrices = []

    for s in fc.pbacktrack(num_backtrack):
        structures.append(s)
        bg, = forgi.load_rna(s)
        pair_tables.append(bg.to_pair_table())
    for pair_idx in pair_tables:
        pair_matrix = np.eye(len(seq))
        pair_idx = np.array(pair_idx[1:]) -1
        for i, j in enumerate(pair_idx):
            if j == -1:  # unpaired
                continue
            else:
                pair_matrix[i, j] += 1
        pair_matrix = pair_matrix / np.sum(pair_matrix, axis=-1, keepdims=True)
        pair_matrices.append(pair_matrix)
    pair_matrix = np.mean(pair_matrices, axis=0)
    return pair_matrix


def positional_signal(length, hidden_size=64,
                      min_timescale=1, max_timescale=50):
    """
    Helper function, constructing basic positional encoding.
    The code is partially based on implementation from Tensor2Tensor library
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """

    if hidden_size % 2 != 0:
        raise ValueError()
    position = np.arange(0, length, dtype=np.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (np.log(np.float32(max_timescale) / np.float32(min_timescale)) / (num_timescales - 1))
    inv_timescales = (
            min_timescale *
            np.exp(np.arange(num_timescales, dtype=np.float32) *
                   -log_timescale_increment))
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return signal


def encode_struct(seq, position_only, add_multiclass):
    position_embedding = positional_signal(len(seq))
    if not position_only:
        _pair_map = create_pairing_matrix(seq)
        position_embedding = np.matmul(_pair_map, position_embedding)
    if add_multiclass:
        multiclass_vector = create_multiclass_vector(seq)
        position_embedding = np.concatenate([position_embedding, multiclass_vector], axis=-1)
    return position_embedding


def get_data(seqs, position_only, add_multiclass):
    max_len = np.max([len(seq) for seq in seqs])
    #     max_len = 41
    padded_seqs = [seq + 'N' * (max_len - len(seq)) for seq in seqs]
    structs_enc = []
    for s in tqdm(padded_seqs):
        structs_enc.append(encode_struct(s, position_only=position_only,
                                         add_multiclass=add_multiclass))
    seq_enc = np.array([np.array(encode_sequence(seq, method='signed')) for seq in tqdm(padded_seqs)])
    structs_enc = np.array(structs_enc)
    return np.concatenate([seq_enc, structs_enc], axis=-1)


def encode_rnacompete_2013(position_only=False, add_multiclass=False):
    if position_only:
        logging.info('Not encoding Secondary Structure')
    if add_multiclass:
        logging.info('Concatenating 5-Class StructureVector')

    filename = 'data/rna_compete_2013_multiclass_{}_position_only_{}.npz'.format(add_multiclass, position_only)
    df = pd.read_csv(dc.Client().get_path('3gsL9R'), compression='gzip')
    rbp_names = [col for col in df.columns if 'RNCMPT' in col]
    df_train = df[df['Fold ID'] == 'A'].copy().reset_index()
    df_test = df[df['Fold ID'] == 'B'].copy().reset_index()
    assert df_train.shape[0] == 120326
    assert df_test.shape[0] == 121031
    cutoffs_train = np.nanpercentile(df_train[rbp_names], 99.95, axis=0)
    cutoffs_test = np.nanpercentile(df_test[rbp_names], 99.95, axis=0)
    y_train = df_train[rbp_names].values
    y_test = df_test[rbp_names].values

    y_train = np.clip(y_train, a_max=cutoffs_train, a_min=None)
    y_test = np.clip(y_test, a_max=cutoffs_test, a_min=None)

    y_train = (y_train - np.nanmean(y_train, axis=0)) / np.nanstd(y_train, axis=0)
    y_test = (y_test - np.nanmean(y_test, axis=0)) / np.nanstd(y_test, axis=0)

    df_train.loc[:, rbp_names] = y_train
    df_test.loc[:, rbp_names] = y_test

    x_train = get_data(df_train.seq, position_only=position_only, add_multiclass=add_multiclass)
    x_test = get_data(df_test.seq, position_only=position_only, add_multiclass=add_multiclass)
    np.savez(filename,
             x=x_train,
             x_test=x_test,
             y=y_train,
             y_test=y_test,
             rbp_names=rbp_names)


def encode_rnacompete_2009(position_only=False, add_multiclass=False):
    if position_only:
        logging.info('Not encoding Secondary Structure')
    if add_multiclass:
        logging.info('Concatenating 5-Class StructureVector')
    filename = 'data/rna_compete_2009_multiclass_{}_position_only_{}.npz'.format(add_multiclass, position_only)
    df = pd.read_csv(dc.Client().get_path('gKXhsM'))
    rbp_names = ['Fusip', 'HuR', 'PTB', 'RBM4', 'SF2', 'SLM2', 'U1A', 'VTS1', 'YB1', ]
    df_train = df[df['Fold ID'] == 'A'].copy().reset_index()
    df_test = df[df['Fold ID'] == 'B'].copy().reset_index()

    assert df_train.shape[0] == 76535
    assert df_test.shape[0] == 76207
    cutoffs_train = np.nanpercentile(df_train[rbp_names], 99.95, axis=0)
    cutoffs_test = np.nanpercentile(df_test[rbp_names], 99.95, axis=0)
    y_train = df_train[rbp_names].values
    y_test = df_test[rbp_names].values

    y_train = np.clip(y_train, a_max=cutoffs_train, a_min=None)
    y_test = np.clip(y_test, a_max=cutoffs_test, a_min=None)

    y_train = (y_train - np.nanmean(y_train, axis=0)) / np.nanstd(y_train, axis=0)
    y_test = (y_test - np.nanmean(y_test, axis=0)) / np.nanstd(y_test, axis=0)

    df_train.loc[:, rbp_names] = y_train
    df_test.loc[:, rbp_names] = y_test

    x_train = get_data(df_train.seq, position_only=position_only, add_multiclass=add_multiclass)
    x_test = get_data(df_test.seq, position_only=position_only, add_multiclass=add_multiclass)
    np.savez(filename,
             x=x_train,
             x_test=x_test,
             y=y_train,
             y_test=y_test,
             rbp_names=rbp_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        default='2013',
                        choices=['2013', '2009'],
                        help="Choice of Dataset to Parse (2013 or 2009)", )
    parser.add_argument("-ns", "--no-structure",
                        action="store_true",
                        help="Do not multiply position embedding with pairing matrix")
    parser.add_argument("-mc", "--multi-class",
                        action="store_true",
                        help="Add average 5 class structure embedding")

    args = parser.parse_args()
    if args.dataset == '2009':
        encode_rnacompete_2009(args.no_structure, args.multi_class)
    elif args.dataset == '2013':
        encode_rnacompete_2013(args.no_structure, args.multi_class)
    else:
        raise ValueError
