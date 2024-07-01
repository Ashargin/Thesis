import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import datetime

from src.cogent.struct.knots import inc_length as cogent_remove_pseudoknots

# import fm
# import torch

# from transformers import AutoTokenizer, AutoModel

# Load RNA-FM model
# rna_fm_model, alphabet = fm.pretrained.rna_fm_t12()
# batch_converter = alphabet.get_batch_converter()
# rna_fm_model.eval()  # disables dropout for deterministic results

# Read motifs
df_motifs = pd.read_csv(Path("resources/motif_seqs.csv"), index_col=0)
df_motifs = df_motifs[df_motifs.time < 0.012].reset_index(drop=True)

# Load DNABERT
# dnabert_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6",
#                                                     trust_remote_code=True)
# dnabert_encoder = AutoModel.from_pretrained("zhihan1996/DNA_bert_6",
#                                                     trust_remote_code=True)


def struct_to_pairs(struct):
    open_brackets = ["(", "[", "<", "{"] + [chr(65 + i) for i in range(26)]
    close_brackets = [")", "]", ">", "}"] + [chr(97 + i) for i in range(26)]

    # Add even more bracket characters in case there are too many pseudoknot levels
    bonus_characters = [chr(i) for i in range(256, 383) if i not in [312, 329, 376]]
    open_brackets += bonus_characters[0::2]
    close_brackets += bonus_characters[1::2]

    opened = [[] for _ in range(len(open_brackets))]
    pairs = {}
    for i, char in enumerate(struct):
        if char == ".":
            pairs[i + 1] = 0
        elif char in open_brackets:
            bracket_type = open_brackets.index(char)
            opened[bracket_type].append(i + 1)
        elif char in close_brackets:
            bracket_type = close_brackets.index(char)
            last_opened = opened[bracket_type].pop()
            pairs[last_opened] = i + 1
            pairs[i + 1] = last_opened
        elif char == "?":
            assert all([c == "?" for c in struct])
            return np.array([0 for i in range(len(struct))])
        else:
            raise Warning("Unknown bracket !")

    pairs = np.array([pairs[i + 1] for i in range(len(struct))])
    return pairs


def pairs_to_struct(pairs):
    open_brackets = ["(", "[", "<", "{"] + [chr(65 + i) for i in range(26)]
    close_brackets = [")", "]", ">", "}"] + [chr(97 + i) for i in range(26)]

    # Add even more bracket characters in case there are too many pseudoknot levels
    bonus_characters = [chr(i) for i in range(256, 383) if i not in [312, 329, 376]]
    open_brackets += bonus_characters[0::2]
    close_brackets += bonus_characters[1::2]

    struct = ["."] * len(pairs)
    bounds = []

    def get_subbounds(new_left, new_right, old_right):
        subbounds = [(new_left, new_right)]
        if new_right < old_right - 2:
            subbounds.append((new_right, old_right))
        return subbounds

    for i, j in enumerate(pairs):
        i += 1
        if (j == 0) or (i > j):
            continue
        if pairs[pairs[i - 1] - 1] != i:  # malformed pairs
            continue

        bracket_type = 0
        while True:
            if len(bounds) == bracket_type:
                bounds.append(get_subbounds(i, j, len(pairs) + 1))
                struct[i - 1] = open_brackets[bracket_type]
                struct[j - 1] = close_brackets[bracket_type]
                break

            # Try to insert (i, j) in bracket_type bounds
            bracket_bounds = bounds[bracket_type]
            for k, b in enumerate(bracket_bounds):
                if (i > b[0]) and (j < b[1]):
                    bounds[bracket_type] += get_subbounds(i, j, b[1])
                    bounds[bracket_type].pop(k)  # remove old outer bound
                    struct[i - 1] = open_brackets[bracket_type]
                    struct[j - 1] = close_brackets[bracket_type]
                    break

            # If failed, look at next bracket_type
            else:
                bracket_type += 1
                continue
            # If succeeded, escape while loop
            break

    return "".join(struct)


def remove_pseudoknots(struct_or_pairs):
    # Wrapper for the pseudoknot removal functions from S. Smit, K. Rother, J. Heringa, and R. Knight
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2248259/
    # https://github.com/pycogent/pycogent/blob/f720cc3753429d130f9e9bc0756b8878c3d50ef2/cogent/struct/knots.py
    seq_format = "struct" if isinstance(struct_or_pairs, str) else "pairs"
    pairs = (
        struct_to_pairs(struct_or_pairs) if seq_format == "struct" else struct_or_pairs
    )

    cogent_pairs = []
    for i, j in enumerate(pairs):
        i += 1
        if (j > 0) and (i < j):
            cogent_pairs.append((i, j))
    cogent_pseudofree_pairs = cogent_remove_pseudoknots(cogent_pairs)
    assert set(cogent_pseudofree_pairs).issubset(cogent_pairs)

    pseudofree_pairs = np.zeros_like(pairs)
    for i, j in cogent_pseudofree_pairs:
        pseudofree_pairs[i - 1] = j
        pseudofree_pairs[j - 1] = i

    pseudofree_struct = pairs_to_struct(pseudofree_pairs)
    assert set(pseudofree_struct).issubset({".", "(", ")"})

    return pseudofree_struct if seq_format == "struct" else pseudofree_pairs


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space
    """
    kmer = [seq[x : x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)

    return kmers


def apply_mutation(seq, struct, mutation_proba=1.0):
    struct_no_pk = re.sub("[^\(\)\.]", ".", struct)
    pairs = struct_to_pairs(struct_no_pk)
    mutations = [
        ("A", "U"),
        ("U", "A"),  # Watson-Crick
        ("G", "C"),
        ("C", "G"),  # Watson-Crick
        ("G", "U"),
        ("U", "G"),  # Wobble
    ]
    mutated_seq = ["" for _ in range(len(seq))]
    for i, j in enumerate(pairs):
        j -= 1
        if j < 0:
            mutated_seq[i] = seq[i]
        elif i < j:
            if np.random.random() < mutation_proba:
                mut_1, mut_2 = mutations[np.random.randint(len(mutations))]
                mutated_seq[i] = mut_1
                mutated_seq[j] = mut_2
            else:
                mutated_seq[i] = seq[i]
                mutated_seq[j] = seq[j]
    mutated_seq = "".join(mutated_seq)

    return mutated_seq, struct


def eval_energy(seq, struct):
    suffix = datetime.datetime.now().strftime("%Y.%m.%d:%H.%M.%S:%f")
    path_in = f"temp_rnaeval_in_{suffix}.txt"
    while os.path.exists(path_in):
        time.sleep(0.01)
        suffix = datetime.datetime.now().strftime("%Y.%m.%d:%H.%M.%S:%f")
        path_in = f"temp_rnaeval_in_{suffix}.txt"
    with open(path_in, "w") as f:
        f.write(f"{seq}\n{struct}")

    output = os.popen(f"RNAeval -i {path_in}").read()
    res = float(output.split(" (")[1].split(")")[0])

    os.remove(path_in)

    return res


def run_preds(
    fnc,
    out_path,
    in_filename="test_sequencewise",
    allow_errors=False,
    use_structs=False,
    max_len=None,
    kwargs={},
    compute_frac=None,
    feed_structs_to_print_fscores=False,
    evaluate_cutting_model=False,
):
    # Read input
    in_path = Path(f"resources/data_structures/{in_filename}.dbn")
    with open(in_path, "r") as f:
        content = f.read()
    lines = content.strip().split("\n")
    assert len(lines) % 3 == 0
    names = lines[0::3]
    seqs = lines[1::3]
    structs = lines[2::3]
    n = len(seqs)

    # Read already predicted
    if evaluate_cutting_model:
        filename, ext = os.path.splitext(out_path)
        out_path = Path(filename + "_cuttingmetrics" + ext)
    if not out_path.exists():
        with open(out_path, "w") as f:
            pass
    with open(out_path, "r") as f:
        processed = f.read()
    lines = processed.split("\n")[1:]
    if lines and not lines[-1]:
        lines = lines[:-1]
    n_processed = len(lines)
    f_out = open(out_path, "w")
    if len(processed) == 0:
        header = (
            "rna_name,seq,struct,break_rate,compression"
            if evaluate_cutting_model
            else "rna_name,seq,struct,pred,ttot,memory"
        )
        f_out.write(f"{header}\n")
    else:
        f_out.write(processed)
    f_out.close()

    def dummy_response(input_len):
        return "?" * input_len, 0.0, 1.0

    # Run
    print(f"Predicting to {out_path}")
    print(f"{n_processed}/{n} already processed")
    skip_counter = 0.0
    for i, (name, seq, struct) in enumerate(zip(names, seqs, structs)):
        if i < n_processed:
            continue

        print(f"{i}/{n}")
        if use_structs:
            kwargs["struct"] = struct
        if feed_structs_to_print_fscores:
            kwargs["struct_to_print_fscores"] = struct
        if evaluate_cutting_model:
            kwargs["return_cuts"] = True

        if compute_frac is not None and skip_counter < 0:
            skip_counter += compute_frac
            pred, ttot, memory = dummy_response(len(seq))
        elif max_len is not None and len(seq) > max_len:
            print(f"Skipping sequence of length {len(seq)}")
            pred, ttot, memory = dummy_response(len(seq))
        elif allow_errors:
            try:
                pred, ttot, memory = fnc(seq, **kwargs)
                if compute_frac is not None:
                    skip_counter += compute_frac - 1
            except (MemoryError, RuntimeError, IndexError, ValueError) as e:
                print(f"Failed for length {len(seq)}. Error: {e}")
                pred, ttot, memory = dummy_response(len(seq))
        else:
            pred, ttot, memory = fnc(seq, **kwargs)
            if compute_frac is not None:
                skip_counter += compute_frac - 1
        if evaluate_cutting_model:
            frags = [p[0] for p in pred]
            struct_no_pseudoknots = re.sub("[^\(\)\.]", ".", struct)
            pairs = struct_to_pairs(struct_no_pseudoknots)
            frag_attrib = np.zeros(
                (
                    len(
                        seq,
                    )
                ),
                dtype=int,
            )
            for i, f in enumerate(frags):
                for start, end in f:
                    frag_attrib[start : end + 1] = i
            n_pairs = 0
            n_breaks = 0
            for i, j in enumerate(pairs):
                j -= 1
                if j > i:
                    n_pairs += 1
                    if frag_attrib[i] != frag_attrib[j]:
                        n_breaks += 1
            break_rate = n_breaks / n_pairs if n_pairs > 0 else 0.0
            compression = (
                1 - ((pd.Series(frag_attrib).value_counts() / len(seq)) ** 2).sum()
            )
            line = f'{name.split("#Name: ")[1]},{seq},{struct},{break_rate},{compression}\n'
        else:
            line = f'{name.split("#Name: ")[1]},{seq},{struct},{pred},{ttot},{memory}\n'
        with open(out_path, "a") as f_out:
            f_out.write(line)


def get_scores(y, y_hat):
    assert len(y) == len(y_hat)
    y_pairs = struct_to_pairs(y)
    y_hat_pairs = struct_to_pairs(y_hat)

    # MXfold2 format
    # https://github.com/mxfold/mxfold2/blob/51b213676708bebd664f0c40873a46e09353e1ee/mxfold2/compbpseq.py#L32
    L = len(y)
    ref = {(i + 1, j) for i, j in enumerate(y_pairs) if i + 1 < j}
    pred = {(i + 1, j) for i, j in enumerate(y_hat_pairs) if i + 1 < j}
    tp = len(ref & pred)
    fp = len(pred - ref)
    fn = len(ref - pred)
    tn = L * (L - 1) // 2 - tp - fp - fn

    this_ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    this_sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    this_fscore = (
        2 * this_sen * this_ppv / (this_sen + this_ppv)
        if (this_ppv + this_sen) > 0
        else 0.0
    )
    this_mcc = (
        (tp * tn - fp * fn)
        / np.sqrt(tp + fp)
        / np.sqrt(tp + fn)
        / np.sqrt(tn + fp)
        / np.sqrt(tn + fn)
        if (tp + fp) > 0 and (tp + fn) > 0 and (tn + fp) > 0 and (tn + fn) > 0
        else 0.0
    )

    return this_ppv, this_sen, this_fscore, this_mcc


def get_scores_df(path_in, without_pseudoknots=False):
    # Read data
    df_preds = pd.read_csv(path_in)
    n = df_preds.shape[0]

    # Compute scores
    fscore = []
    struct_nopk = []
    pred_nopk = []
    fscore_nopk = []
    for i, (y, y_hat) in enumerate(zip(df_preds.struct, df_preds.pred)):
        if n >= 10 and i % int(n / 10) == 0:
            print(f"{10 * int(i / int(n / 10))}%")

        _, _, this_fscore, _ = get_scores(y, y_hat)
        fscore.append(this_fscore)

        if without_pseudoknots:
            y_nopk = remove_pseudoknots(y)
            y_hat_nopk = remove_pseudoknots(y_hat)
            _, _, this_fscore_nopk, _ = get_scores(y_nopk, y_hat_nopk)
            struct_nopk.append(y_nopk)
            pred_nopk.append(y_hat_nopk)
            fscore_nopk.append(this_fscore_nopk)
        else:
            struct_nopk.append(np.nan)
            pred_nopk.append(np.nan)
            fscore_nopk.append(np.nan)

    # Create dataframe
    skipped = np.array(["?" in p for p in df_preds.pred])
    data = pd.DataFrame(
        {
            "rna_name": df_preds.rna_name,
            "seq": df_preds.seq,
            "struct": df_preds.struct,
            "struct_nopk": struct_nopk,
            "pred": df_preds.pred,
            "pred_nopk": pred_nopk,
            "length": df_preds.seq.apply(len),
            "fscore": fscore,
            "fscore_nopk": fscore_nopk,
            "time": df_preds.ttot,
            "memory": df_preds.memory,
        }
    )
    if not without_pseudoknots:
        data.drop(["struct_nopk", "pred_nopk", "fscore_nopk"], axis=1, inplace=True)

    cutting_metric_filename = (
        path_in.name.replace("_mx_", "_")
        .replace("_rnaf_", "_")
        .replace("_lf_", "_")
        .replace("_kf_", "_")
        .replace("_sub_", "_")
        .replace("_ens_", "_")
        .replace(".csv", "_cuttingmetrics.csv")
    )
    cutting_metric_path = (
        path_in.parents[2]
        / "cutting_metrics"
        / path_in.parent.name
        / cutting_metric_filename
    )
    if cutting_metric_path.exists():
        df_cutting_metrics = pd.read_csv(cutting_metric_path)
        assert np.all(data.rna_name == df_cutting_metrics.rna_name)
        data["cut_break_rate"] = df_cutting_metrics.break_rate
        data["cut_compression"] = df_cutting_metrics.compression

    data.time = data.time.astype(float)
    data.memory = data.memory.astype(float)
    data = data.iloc[~skipped, :]

    # if n < 10:
    #     return data
    # ax = sns.kdeplot(data=data, x="length")
    # x, y = ax.get_lines()[-1].get_data()
    #
    # def inverse_density(length):
    #     upper_x_bound = (x <= length).argmin()
    #     lower_x, upper_x = x[upper_x_bound - 1], x[upper_x_bound]
    #     lower_y, upper_y = y[upper_x_bound - 1], y[upper_x_bound]
    #     perc = (length - lower_x) / (upper_x - lower_x)
    #     val = lower_y + perc * (upper_y - lower_y)
    #     return 1 / val
    #
    # data["weight"] = data.length.apply(inverse_density)
    # data.weight /= data.weight.mean()
    # plt.close()

    return data


def seq_to_one_hot(seq):
    seq_array = np.array(list(seq))
    seq_one_hot = np.zeros((4, len(seq)))
    for i, var in enumerate(["A", "U", "C", "G"]):
        seq_one_hot[i, seq_array == var] = 1

    return seq_one_hot.T


def seq_to_motif_matches(seq, max_motifs=None, **kwargs):
    if max_motifs == 0:
        return np.zeros((len(seq), max_motifs))

    def get_motif_matches(seq, motif, overlap=True, stack=True, normalize=False):
        # Compile regex with pattern
        pattern = "(" + motif.replace("*", ").*?(") + ")"
        if overlap:
            pattern = "(?=" + pattern + ")"
        regex = re.compile(pattern)

        # Iterate over matches and get group spans
        n_groups = motif.count("*") + 1
        scores = np.zeros(len(seq))
        for match in regex.finditer(seq):
            for i in range(n_groups):
                start, end = match.span(i + 1)
                scores[start:end] += 1

        if not stack:
            scores = 1.0 * (scores > 0)
        elif normalize and scores.max() > 0:
            scores /= scores.max()

        return scores

    max_motifs = df_motifs.shape[0] if max_motifs is None else max_motifs
    used_index = (
        df_motifs.sort_index().sort_values("time").index[:max_motifs].sort_values()
    )
    df_motifs_used = df_motifs.loc[used_index]
    scores_matrix = df_motifs_used.motif_seq.apply(
        lambda motif: get_motif_matches(seq, motif, **kwargs)
    )
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


# def seq_to_dnabert(seq):
#     tokenized = dnabert_tokenizer(
#         seq2kmer(seq.replace("U", "T").replace("~", "M"), k=6),
#         padding="longest",
#         pad_to_multiple_of=512,
#     )
#     encoded = dnabert_encoder(
#         torch.tensor([tokenized["input_ids"]]).view(-1, 512),
#         torch.tensor([tokenized["attention_mask"]]).view(-1, 512),
#     )
#     seq_mat, pooled_seq_mat = encoded[0], encoded[1]
#
#     tokens_len = len(seq) - 3
#     seq_mat = np.vstack(seq_mat.detach().numpy())[:tokens_len]
#     seq_mat = np.vstack(
#         [seq_mat[0], seq_mat[0], seq_mat, seq_mat[-1]]
#     )  ###### fix size ?
#     # pooled_seq_mat = np.mean(pooled_seq_mat.detach().numpy(), axis=0)
#
#     return seq_mat


def format_data(seq, cuts=None, input_format="motifs", max_motifs=None, **kwargs):
    seq_array = None
    if input_format == "one_hot":
        seq_array = seq_to_one_hot(seq)

    elif input_format == "motifs":
        seq_one_hot = seq_to_one_hot(seq)
        seq_motifs = seq_to_motif_matches(seq, max_motifs=max_motifs, **kwargs)
        seq_array = np.hstack([seq_one_hot, seq_motifs])

    # elif input_format == 'rna_fm':
    #     seq_array = seq_to_rna_fm(seq)

    elif input_format == "dnabert":
        seq_array = seq_to_dnabert(seq)

    elif input_format == "concatenate":
        seq_one_hot = seq_to_one_hot(seq)
        seq_motifs = seq_to_motif_matches(seq, max_motifs=max_motifs, **kwargs)
        # seq_rnafm = seq_to_rna_fm(seq)
        seq_dnabert = seq_to_dnabert(seq)
        # seq_array = np.hstack([seq_one_hot, seq_motifs, seq_rnafm, seq_dnabert])
        seq_array = np.hstack([seq_one_hot, seq_motifs, seq_dnabert])

    if cuts is not None:
        cuts_ints = [int(c) for c in cuts[1:-1].split()]
        cuts_array = np.zeros(len(seq))
        cuts_array[cuts_ints] = 1
        return seq_array, cuts_array

    return seq_array
