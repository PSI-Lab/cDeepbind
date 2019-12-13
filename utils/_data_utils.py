from modelzoo.utils import encode_sequence
import numpy as np
from tqdm.autonotebook import tqdm
import rnafoldr


def encode_struct(seq):
    return rnafoldr.fold_unpair(seq.replace('T', 'U'),
                                winsize=len(seq), span=len(seq))


def get_data(seqs, context_size=200):
    #     max_len = np.max([len(seq) for seq in seqs])
    max_len = 41
    padded_seqs = [seq + 'N' * (max_len - len(seq)) for seq in seqs]
    #     structs_enc = np.expand_dims(np.array([encode_struct(seq) for seq in tqdm(padded_seqs)]), axis=2)
    seq_enc = np.array([np.array(encode_sequence(seq, method='signed')) for seq in tqdm(padded_seqs)])
    return seq_enc
#     return np.concatenate([seq_enc, structs_enc], axis=-1)
