import os
import numpy as np
import pandas as pd
import re
import seaborn as sns
# import fm
import torch

# Load RNA-FM model
# rna_fm_model, alphabet = fm.pretrained.rna_fm_t12()
# batch_converter = alphabet.get_batch_converter()
# rna_fm_model.eval()  # disables dropout for deterministic results

# Read motifs
df_motifs = pd.read_csv(os.path.join('resources', 'motif_seqs.csv'), index_col=0)
df_motifs = df_motifs[df_motifs.time < 0.012]


def struct_to_pairs(y):
    open_brackets = ['(', '[', '<', '{', 'A', 'B', 'C']
    close_brackets = [')', ']', '>', '}', 'a', 'b', 'c']
    opened = [[] for _ in range(len(open_brackets))]
    pairs = {}
    for i, char in enumerate(y):
        if char == '.':
            pairs[i+1] = 0
        elif char in open_brackets:
            bracket_type = open_brackets.index(char)
            opened[bracket_type].append(i+1)
        elif char in close_brackets:
            bracket_type = close_brackets.index(char)
            last_opened = opened[bracket_type].pop()
            pairs[last_opened] = i+1
            pairs[i+1] = last_opened
        elif char == '?':
            assert all([c == '?' for c in y])
            return {i+1: 0 for i in range(len(y))}
        else:
            raise Warning('Unknown bracket !')

    pairs = np.array([pairs[i+1] for i in range(len(y))])
    return pairs


def run_preds(fnc, out_path, in_path='bpRNA_1M/dbnFiles/allDbn.dbn', allow_errors=False,
                                                                     use_structs=False,
                                                                     store_cuts=False):
    # Read input
    with open(in_path, 'r') as f:
        content = f.read()
    lines = content.split('\n')
    assert len(lines) % 3 == 0
    headers = lines[0::3]
    seqs = lines[1::3]
    structs = lines[2::3]
    n = len(seqs)

    # Read already predicted
    if store_cuts:
        filename, ext = os.path.splitext(out_path)
        out_path = filename + '_cuts' + ext
    if not os.path.exists(out_path):
        with open(out_path, 'w') as f:
            pass
    if not store_cuts:
        with open(out_path, 'r') as f:
            processed = f.read()
        lines = processed.split('\n')[1:]
        if lines and not lines[-1]:
            lines = lines[:-1]
        n_processed = len(lines)
        f_out = open(out_path, 'w')
        if len(processed) == 0:
            f_out.write('rna_name,seq,struct,pred,ttot,memory\n')
        f_out.write(processed)
    else:
        f_out = open(out_path, 'w')
        f_out.write('rna_name,seq,cuts,outer\n')
        n_processed = 0

    # Run
    print(f'{n_processed}/{n} already processed')
    for i, (header, seq, struct) in enumerate(zip(headers, seqs, structs)):
        if i < n_processed:
            continue

        print(f'{i}/{n}')
        kwargs = {}
        if use_structs:
            kwargs['struct'] = struct
        if store_cuts:
            kwargs['cuts_file'] = f_out
            kwargs['rna_name'] = header
        if allow_errors:
            try:
                pred, _, _, ttot, memory = fnc(seq, **kwargs)
            except (RuntimeError, IndexError, ValueError) as e:
                print(f'Failed: length {len(seq)}, error {e}')
                pred = '?' * len(seq)
                ttot = 0.
                memory = 1.
        else:
            pred, _, _, ttot, memory = fnc(seq, **kwargs)
        if not store_cuts:
            line = f'{header.split("#Name: ")[1]},{seq},{struct},{pred},{ttot},{memory}\n'
            f_out.write(line)
    f_out.close()


def get_scores_df(preds_path):
    # Read data
    df_preds = pd.read_csv(preds_path)
    n = df_preds.shape[0]

    # Compute scores
    ppv = []
    sen = []
    fscore = []
    for i, (y, y_hat) in enumerate(zip(df_preds.struct, df_preds.pred)):
        if i % int(n / 10) == 0:
            print(f'{10 * int(i / int(n / 10))}%')

        assert len(y) == len(y_hat)
        y_pairs = struct_to_pairs(y)
        y_hat_pairs = struct_to_pairs(y_hat)

        tp = np.sum((y_pairs == y_hat_pairs) & (y_hat_pairs != 0))
        fp = np.sum((y_pairs != y_hat_pairs) & (y_hat_pairs != 0))
        fn = np.sum((y_pairs != y_hat_pairs) & (y_hat_pairs == 0))

        this_ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        this_sen = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        this_fscore = 2 * this_sen * this_ppv / (this_sen + this_ppv) \
                                if (this_ppv + this_sen) > 0 else np.nan
        ppv.append(this_ppv)
        sen.append(this_sen)
        fscore.append(this_fscore)

    # Create dataframe
    skipped = np.array(['?' in p for p in df_preds.pred])
    data = pd.DataFrame({'rna_name': df_preds.rna_name,
                         'seq': df_preds.seq,
                         'struct': df_preds.struct,
                         'pred': df_preds.pred,
                         'length': df_preds.seq.apply(len),
                         'ppv': ppv,
                         'sen': sen,
                         'fscore': fscore,
                         'time': df_preds.ttot,
                         'memory': df_preds.memory})

    data.time = data.time.astype(float)
    data.memory = data.memory.astype(float)
    data = data.iloc[~skipped, :]

    ax = sns.kdeplot(data=data, x='length')
    x, y = ax.get_lines()[-1].get_data()

    def inverse_density(length):
        upper_x_bound = (x <= length).argmin()
        lower_x, upper_x = x[upper_x_bound-1], x[upper_x_bound]
        lower_y, upper_y = y[upper_x_bound-1], y[upper_x_bound]
        perc = (length - lower_x) / (upper_x - lower_x)
        val = lower_y + perc * (upper_y - lower_y)
        return 1 / val

    data['weight'] = data.length.apply(inverse_density)
    data.weight /= data.weight.mean()

    return data


def seq_to_one_hot(seq):
    seq_array = np.array(list(seq))
    seq_one_hot = np.zeros((4, len(seq)))
    for i, var in enumerate(['A', 'U', 'C', 'G']):
        seq_one_hot[i, seq_array == var] = 1

    return seq_one_hot.T


def seq_to_motif_matches(seq, **kwargs):

    def get_motif_matches(seq, motif, overlap=True, stack=True, normalize=False):
        # Compile regex with pattern
        pattern = '(' + motif.replace('*', ').*?(') + ')'
        if overlap:
            pattern = '(?=' + pattern + ')'
        regex = re.compile(pattern)

        # Iterate over matches and get group spans
        n_groups = motif.count('*') + 1
        scores = np.zeros(len(seq))
        for match in regex.finditer(seq):
            for i in range(n_groups):
                start, end = match.span(i+1)
                scores[start:end] += 1

        if not stack:
            scores = 1. * (scores > 0)
        elif normalize and scores.max() > 0:
            scores /= scores.max()

        return scores

    scores_matrix = df_motifs.motif_seq.apply(lambda motif: get_motif_matches
                                                        (seq, motif, **kwargs))
    scores_matrix = np.vstack(scores_matrix)

    return scores_matrix.T


# def seq_to_rna_fm(seq):
#     data = [('seq_name', seq)]
#     _, _, batch_tokens = batch_converter(data)
#     with torch.no_grad():
#         results = rna_fm_model(batch_tokens, repr_layers=[12])
#     token_embeddings = results['representations'][12].numpy()
#     token_embeddings = token_embeddings.reshape((len(seq) + 2, 640))[1:-1, :]
#
#     return token_embeddings


def format_data(seq, cuts=None, input_format='concatenate', **kwargs):
    seq_array = None
    if input_format == 'one_hot':
        seq_array = seq_to_one_hot(seq)

    elif input_format == 'motifs':
        seq_array = seq_to_motif_matches(seq, **kwargs)

    # elif input_format == 'rna_fm':
    #     seq_array = seq_to_rna_fm(seq)

    elif input_format == 'concatenate':
        seq_one_hot = seq_to_one_hot(seq)
        seq_motifs = seq_to_motif_matches(seq, **kwargs)
        # seq_rnafm = seq_to_rna_fm(seq)
        # seq_array = np.hstack([seq_one_hot, seq_motifs, seq_rnafm])
        seq_array = np.hstack([seq_one_hot, seq_motifs])

    if cuts is not None:
        cuts_ints = [int(c) for c in cuts[1:-1].split()]
        cuts_array = np.zeros(len(seq))
        cuts_array[cuts_ints] = 1
        return seq_array, cuts_array

    return seq_array
