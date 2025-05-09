import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import warnings

from src.cogent.struct.rna2d import Pairs as CogentPairs
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
            try:
                last_opened = opened[bracket_type].pop()
            except IndexError:
                raise ValueError(
                    "Malformed structure was given (closing bracket appears before corresponding opening bracket)."
                )
            pairs[last_opened] = i + 1
            pairs[i + 1] = last_opened
        elif char == "?":
            assert all([c == "?" for c in struct])
            return np.array([0 for i in range(len(struct))])
        else:
            raise ValueError(
                "Malformed structure was given (unknown bracket character)."
            )

    try:
        pairs = np.array([pairs[i + 1] for i in range(len(struct))])
        return pairs
    except KeyError:
        raise ValueError(
            "Malformed structure was given (opening bracket has no corresponding closing bracket)."
        )


def pairs_to_struct(pairs):
    pseudofree_pairs, pseudoknot_pairs = remove_pseudoknots(
        pairs, return_pseudoknots=True
    )
    stacked = np.vstack([pseudofree_pairs, pseudoknot_pairs])
    assert stacked.min(axis=0).max() == 0

    pseudofree_struct = _sub_pairs_to_struct(pseudofree_pairs)
    pseudoknot_struct = _sub_pairs_to_struct(pseudoknot_pairs, start_bracket=1)

    struct = "".join(
        [
            db1 if db2 == "." else db2
            for db1, db2 in zip(pseudofree_struct, pseudoknot_struct)
        ]
    )

    return struct


def optimize_pseudoknots(struct):
    return pairs_to_struct(struct_to_pairs(struct))


def _sub_pairs_to_struct(pairs, start_bracket=0):
    open_brackets = ["(", "[", "<", "{"] + [chr(65 + i) for i in range(26)]
    close_brackets = [")", "]", ">", "}"] + [chr(97 + i) for i in range(26)]
    open_brackets = open_brackets[start_bracket:]
    close_brackets = close_brackets[start_bracket:]

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
        if (j == 0) or (i >= j):
            continue
        if pairs[j - 1] != i:  # malformed pairs
            warnings.warn(
                "Some malformed pairs were given (non-symmetrical pairs) and will be ignored when converting to dot-bracket format."
            )
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


def remove_pseudoknots(struct_or_pairs, return_pseudoknots=False):
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
        if (j == 0) or (i >= j):
            continue
        if pairs[j - 1] != i:  # malformed pairs
            warnings.warn(
                "Some malformed pairs were given (non-symmetrical pairs) and will be ignored when converting to dot-bracket format."
            )
            continue
        cogent_pairs.append((i, j))
    cogent_pairs = CogentPairs(cogent_pairs)

    # Call the pseudoknot removal function
    cogent_pseudofree_pairs, cogent_pseudoknot_pairs = cogent_remove_pseudoknots(
        cogent_pairs, return_removed=True
    )
    assert set(cogent_pseudofree_pairs).issubset(cogent_pairs)
    assert set(cogent_pseudoknot_pairs) == set(cogent_pairs) - set(
        cogent_pseudofree_pairs
    )

    def cogent_to_numpy(cogent_pairs):
        numpy_pairs = np.zeros_like(pairs)
        for i, j in cogent_pairs:
            numpy_pairs[i - 1] = j
            numpy_pairs[j - 1] = i
        return numpy_pairs

    pseudofree_pairs = cogent_to_numpy(cogent_pseudofree_pairs)
    pseudoknot_pairs = cogent_to_numpy(cogent_pseudoknot_pairs)

    if seq_format == "pairs":
        return (
            (pseudofree_pairs, pseudoknot_pairs)
            if return_pseudoknots
            else pseudofree_pairs
        )

    pseudofree_struct = _sub_pairs_to_struct(pseudofree_pairs)
    pseudoknot_struct = _sub_pairs_to_struct(pseudoknot_pairs, start_bracket=1)

    return (
        (pseudofree_struct, pseudoknot_struct)
        if return_pseudoknots
        else pseudofree_struct
    )


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


def apply_mutation(seq, struct, mutation_proba=1.0, struct_deletion_proba=0.0):
    struct_no_pk = re.sub(r"[^\(\)\.]", ".", struct)
    pairs = struct_to_pairs(struct_no_pk)

    # Sequence mutation
    mutations = [
        ("A", "U"),
        ("U", "A"),  # Watson-Crick
        ("G", "C"),
        ("C", "G"),  # Watson-Crick
        ("G", "U"),
        ("U", "G"),  # Wobble
    ]
    augmented_seq = ["" for _ in range(len(seq))]
    for i, j in enumerate(pairs):
        j -= 1
        if j < 0:
            augmented_seq[i] = seq[i]
        elif i < j:
            if np.random.random() < mutation_proba:
                mut_1, mut_2 = mutations[np.random.randint(len(mutations))]
                augmented_seq[i] = mut_1
                augmented_seq[j] = mut_2
            else:
                augmented_seq[i] = seq[i]
                augmented_seq[j] = seq[j]
    augmented_seq = "".join(augmented_seq)

    # Structure deletion
    for i, j in enumerate(pairs):
        j -= 1
        if i < j:
            if np.random.random() < struct_deletion_proba:
                pairs[i] = 0
                pairs[j] = 0
    augmented_struct = pairs_to_struct(pairs)
    augmented_struct = list(augmented_struct)
    for i, c in enumerate(struct):
        if c not in [".", "(", ")"]:
            augmented_struct[i] = c
    augmented_struct = "".join(augmented_struct)

    return augmented_seq, augmented_struct


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
            "rna_name,seq,struct,break_rate,compression,frags,ttot"
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
        struct = optimize_pseudoknots(struct)
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
            except (
                MemoryError,
                RuntimeError,
                IndexError,
                ValueError,
                TimeoutError,
            ) as e:
                print(f"Failed for length {len(seq)}. Error: {e}")
                pred, ttot, memory = dummy_response(len(seq))
        else:
            pred, ttot, memory = fnc(seq, **kwargs)
            if compute_frac is not None:
                skip_counter += compute_frac - 1
        if evaluate_cutting_model:
            frags = [p[0] for p in pred]
            struct_no_pseudoknots = re.sub(r"[^\(\)\.]", ".", struct)
            pairs = struct_to_pairs(struct_no_pseudoknots)
            frag_attrib = np.zeros(len(seq), dtype=int)
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
            frags_txt = "["
            for f in frags:
                frags_txt += "["
                for i, j in f:
                    frags_txt += f"[{i} {j}]"
                frags_txt += "]"
            frags_txt += "]"
            line = f'{name.split("#Name: ")[1]},{seq},{struct},{break_rate},{compression},{frags_txt},{ttot}\n'
        else:
            pred = optimize_pseudoknots(pred) if set(pred) != {"?"} else pred
            line = f'{name.split("#Name: ")[1]},{seq},{struct},{pred},{ttot},{memory}\n'
        with open(out_path, "a") as f_out:
            f_out.write(line)


def _confusion_matrix_to_scores(tp, fp, fn, tn):
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fscore = 2 * sen * ppv / (sen + ppv) if (ppv + sen) > 0 else 0.0
    mcc = (
        (
            (tp * tn - fp * fn)
            / np.sqrt(tp + fp)
            / np.sqrt(tp + fn)
            / np.sqrt(tn + fp)
            / np.sqrt(tn + fn)
            if (tp + fp) > 0 and (tp + fn) > 0 and (tn + fp) > 0 and (tn + fn) > 0
            else 0.0
        )
        if tn is not None
        else None
    )
    return ppv, sen, fscore, mcc


def get_structure_scores(y, yhat):
    y_pairs = struct_to_pairs(y)
    yhat_pairs = struct_to_pairs(yhat)

    # MXfold2 definition of true/false positives/negatives
    # https://github.com/mxfold/mxfold2/blob/51b213676708bebd664f0c40873a46e09353e1ee/mxfold2/compbpseq.py#L32
    L = len(y)
    ref = {(i + 1, j) for i, j in enumerate(y_pairs) if i + 1 < j}
    pred = {(i + 1, j) for i, j in enumerate(yhat_pairs) if i + 1 < j}

    tp = len(ref & pred)
    fp = len(pred - ref)
    fn = len(ref - pred)
    tn = L * (L - 1) // 2 - tp - fp - fn

    return _confusion_matrix_to_scores(tp, fp, fn, tn)


def get_pseudoknot_interaction_scores(y, yhat):
    y_pairs = struct_to_pairs(y)
    yhat_pairs = struct_to_pairs(yhat)
    y_cogent_pairs = [(i + 1, j) for i, j in enumerate(y_pairs) if j > i + 1]
    yhat_cogent_pairs = [(i + 1, j) for i, j in enumerate(yhat_pairs) if j > i + 1]

    def fast_pk_interaction_search(cogent_pairs):
        pks = set()
        for idx, (i, j) in enumerate(cogent_pairs):
            for (k, l) in cogent_pairs[idx + 1 :]:
                if k > j:
                    break
                if l > j:  # such that i < k < j < l by construction, ie a pseudoknot
                    pks.add(((i, j), (k, l)))
        return pks

    L = len(y)
    ref = fast_pk_interaction_search(y_cogent_pairs)
    pred = fast_pk_interaction_search(yhat_cogent_pairs)

    # Similarly as in get_structure_scores
    # The number of possible pseudoknots is L * (L - 1) * (L - 2) * (L - 3) / 24
    tp = len(ref & pred)
    fp = len(pred - ref)
    fn = len(ref - pred)
    tn = L * (L - 1) * (L - 2) * (L - 3) // 24 - tp - fp - fn

    return _confusion_matrix_to_scores(tp, fp, fn, tn)


def get_pseudoknot_motif_scores(
    y, yhat, return_confusion_matrix=False, min_pairs=1, min_distance_in_pk=6
):
    y_pairs = struct_to_pairs(y)
    yhat_pairs = struct_to_pairs(yhat)
    y_cogent_pairs = [(i + 1, j) for i, j in enumerate(y_pairs) if j > i + 1]
    yhat_cogent_pairs = [(i + 1, j) for i, j in enumerate(yhat_pairs) if j > i + 1]

    def fast_pk_motif_search(cogent_pairs):
        left_links = {}
        for idx, (i, j) in enumerate(cogent_pairs):
            for (k, l) in cogent_pairs[idx + 1 :]:
                if k > j:
                    break
                if l > j:  # such that i < k < j < l by construction, ie a pseudoknot
                    if (i, j) in left_links:
                        left_links[(i, j)].append((k, l))
                    else:
                        left_links[(i, j)] = [(k, l)]
        pks = []
        for left_p, this_right_bound in left_links.items():
            for left_bound, right_bound in pks:
                if right_bound == this_right_bound:
                    left_bound.append(left_p)
                    break
            else:
                pks.append(([left_p], this_right_bound))

        return pks

    ref = fast_pk_motif_search(y_cogent_pairs)
    pred = fast_pk_motif_search(yhat_cogent_pairs)

    def get_distance_in_pseudoknot(leftb, rightb):
        leftb = list(sorted(leftb, key=lambda x: x[0]))
        rightb = list(sorted(rightb, key=lambda x: x[0]))
        # left_distance = rightb[-1][0] - leftb[0][0] - len(leftb) - len(rightb) + 1
        left_distance = rightb[-1][0] - leftb[0][0] - 1

        leftb = list(sorted(leftb, key=lambda x: x[1]))
        rightb = list(sorted(rightb, key=lambda x: x[1]))
        # right_distance = rightb[-1][1] - leftb[0][1] - len(leftb) - len(rightb) + 1
        right_distance = rightb[-1][1] - leftb[0][1] - 1

        return left_distance + right_distance

    ref = [
        (leftb, rightb)
        for leftb, rightb in ref
        if get_distance_in_pseudoknot(leftb, rightb) >= min_distance_in_pk
    ]
    pred = [
        (leftb, rightb)
        for leftb, rightb in pred
        if get_distance_in_pseudoknot(leftb, rightb) >= min_distance_in_pk
    ]

    # Compute tp / fp / fn
    positives = [False for _ in range(len(ref))]
    fp = 0
    for pred_leftb, pred_rightb in pred:
        is_true = False
        for i, (ref_leftb, ref_rightb) in enumerate(ref):
            left_found = len(set(pred_leftb) & (set(ref_leftb)))
            if left_found >= min_pairs:
                right_found = len(set(pred_rightb) & (set(ref_rightb)))
                if right_found >= min_pairs:
                    positives[i] = True
                    is_true = True
        if not is_true:
            fp += 1
    tp = sum(positives)
    fn = len(ref) - tp
    tn = None

    if return_confusion_matrix:
        return _confusion_matrix_to_scores(tp, fp, fn, tn), (tp, fp, fn, tn)
    return _confusion_matrix_to_scores(tp, fp, fn, tn)


def get_scores_df(path_in):
    # Read data
    path_in = Path(path_in)
    df_preds = pd.read_csv(path_in)
    n = df_preds.shape[0]

    # Compute scores
    ppvs = []
    sens = []
    fscores = []
    mccs = []
    ppvs_nopk = []
    sens_nopk = []
    fscores_nopk = []
    mccs_nopk = []
    pk_motif_ppvs = []
    pk_motif_sens = []
    pk_motif_fscores = []
    pk_motif_mccs = []
    pk_motif_tps = []
    pk_motif_fps = []
    pk_motif_fns = []
    pk_interact_ppvs = []
    pk_interact_sens = []
    pk_interact_fscores = []
    pk_interact_mccs = []
    print(f"Processing {path_in.name}...")
    for i, (y, yhat) in enumerate(zip(df_preds.struct, df_preds.pred)):
        if n >= 10 and i % int(n / 10) == 0:
            print(f"{10 * int(i / int(n / 10))}%")

        ppv, sen, fscore, mcc = get_structure_scores(y, yhat)
        ppvs.append(ppv)
        sens.append(sen)
        fscores.append(fscore)
        mccs.append(mcc)

        ppv, sen, fscore, mcc = get_structure_scores(
            re.sub(r"[^\(\)\.]", ".", y), re.sub(r"[^\(\)\.]", ".", yhat)
        )
        ppvs_nopk.append(ppv)
        sens_nopk.append(sen)
        fscores_nopk.append(fscore)
        mccs_nopk.append(mcc)

        (ppv, sen, fscore, mcc), (tp, fp, fn, _) = get_pseudoknot_motif_scores(
            y, yhat, return_confusion_matrix=True
        )
        pk_motif_ppvs.append(ppv)
        pk_motif_sens.append(sen)
        pk_motif_fscores.append(fscore)
        pk_motif_mccs.append(mcc)
        pk_motif_tps.append(tp)
        pk_motif_fps.append(fp)
        pk_motif_fns.append(fn)

        ppv, sen, fscore, mcc = get_pseudoknot_interaction_scores(y, yhat)
        pk_interact_ppvs.append(ppv)
        pk_interact_sens.append(sen)
        pk_interact_fscores.append(fscore)
        pk_interact_mccs.append(mcc)

    # Create dataframe
    skipped = np.array(["?" in p for p in df_preds.pred])
    data = pd.DataFrame(
        {
            "rna_name": df_preds.rna_name,
            "seq": df_preds.seq,
            "struct": df_preds.struct,
            "pred": df_preds.pred,
            "length": df_preds.seq.apply(len),
            "ppv": ppvs,
            "sen": sens,
            "fscore": fscores,
            "mcc": mccs,
            "ppv_nopk": ppvs_nopk,
            "sen_nopk": sens_nopk,
            "fscore_nopk": fscores_nopk,
            "mcc_nopk": mccs_nopk,
            "pk_motif_ppv": pk_motif_ppvs,
            "pk_motif_sen": pk_motif_sens,
            "pk_motif_fscore": pk_motif_fscores,
            "pk_motif_mcc": pk_motif_mccs,
            "pk_motif_tp": pk_motif_tps,
            "pk_motif_fn": pk_motif_fns,
            "pk_motif_fp": pk_motif_fps,
            "pk_interact_ppv": pk_interact_ppvs,
            "pk_interact_sen": pk_interact_sens,
            "pk_interact_fscore": pk_interact_fscores,
            "pk_interact_mcc": pk_interact_mccs,
            "time": df_preds.ttot,
            "memory": df_preds.memory,
        }
    )

    cutting_metric_filename = (
        path_in.name.replace("_mx_", "_")
        .replace("_rnaf_", "_")
        .replace("_lf_", "_")
        .replace("_kf_", "_")
        .replace("_ipk_", "_")
        .replace("_pbk_", "_")
        .replace("_pks_", "_")
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
        data["cut_time"] = df_cutting_metrics.ttot

    data = data.iloc[~skipped, :]

    print("Done.")

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


def augment_deletion(seq, struct, min_len=1, max_len=20):
    max_len = min(max_len, len(seq) - 1)
    min_len = min(min_len, max_len)

    # Pick deletion length and index
    delete_len = np.random.randint(min_len, max_len + 1)
    delete_ind = np.random.randint(len(seq) - delete_len + 1)
    augmented_seq = seq[:delete_ind] + seq[delete_ind + delete_len :]

    # Adjust structure, remove base pairs for deleted nucleotides
    pairs = struct_to_pairs(struct)
    augmented_pairs = np.concatenate(
        [
            pairs[:delete_ind],
            pairs[delete_ind + delete_len :],
        ]
    ).astype(int)

    def translate(j):
        if j <= delete_ind:
            return j
        elif j <= delete_ind + delete_len:
            return 0
        else:
            return j - delete_len

    augmented_pairs = np.array([translate(j) if j > 0 else j for j in augmented_pairs])
    augmented_struct = pairs_to_struct(augmented_pairs)

    return augmented_seq, augmented_struct


def augment_insertion(seq, struct, min_len=1, max_len=20):
    min_len = min(min_len, max_len)

    # Pick insertion length and index, sample random inserted RNA
    insert_len = np.random.randint(min_len, max_len + 1)
    insert_ind = np.random.randint(len(seq) + 1)
    insertion = "".join(np.random.choice(["A", "U", "C", "G"], insert_len))
    augmented_seq = seq[:insert_ind] + insertion + seq[insert_ind:]

    # Adjust structure, set no base pairs for inserted nucleotides
    augmented_struct = struct[:insert_ind] + "." * insert_len + struct[insert_ind:]

    return augmented_seq, augmented_struct


def augment_translocation(seq, struct, min_len=1, max_len=20):
    max_len = min(max_len, len(seq))
    min_len = min(min_len, max_len)

    # Pick translocation length
    shift_len = np.random.randint(min_len, max_len + 1)
    if np.random.random() < 0.5:
        shift_len *= -1
    augmented_seq = seq[-shift_len:] + seq[:-shift_len]

    # Adjust structure
    pairs = struct_to_pairs(struct)
    augmented_pairs = np.concatenate(
        [
            pairs[-shift_len:],
            pairs[:-shift_len],
        ]
    ).astype(int)

    def translate(j):
        j = (j + shift_len) % len(seq)
        if j == 0:
            j = len(seq)
        return j

    augmented_pairs = np.array([translate(j) if j > 0 else j for j in augmented_pairs])
    augmented_struct = pairs_to_struct(augmented_pairs)

    return augmented_seq, augmented_struct


def augment_inversion(seq, struct, min_len=2, max_len=20):
    max_len = min(max_len, len(seq))
    min_len = min(min_len, max_len)

    # Pick inversion length and index
    inversion_len = np.random.randint(min_len, max_len + 1)
    inversion_ind = np.random.randint(len(seq) - inversion_len + 1)
    augmented_seq = (
        seq[:inversion_ind]
        + seq[inversion_ind : inversion_ind + inversion_len][::-1]
        + seq[inversion_ind + inversion_len :]
    )

    # Adjust structure, remove base pairs between inversed nucleotides and other nucleotides
    pairs = struct_to_pairs(struct)
    augmented_pairs = np.concatenate(
        [
            pairs[:inversion_ind],
            pairs[inversion_ind : inversion_ind + inversion_len][::-1],
            pairs[inversion_ind + inversion_len :],
        ]
    ).astype(int)

    def translate(i, j):
        from_inversed = (i + 1 > inversion_ind) and (
            i + 1 <= inversion_ind + inversion_len
        )
        to_inversed = (j > inversion_ind) and (j <= inversion_ind + inversion_len)
        if (from_inversed and not to_inversed) or (not from_inversed and to_inversed):
            return 0

        if j <= inversion_ind:
            return j
        elif j <= inversion_ind + inversion_len:
            return 2 * inversion_ind + inversion_len + 1 - j
        else:
            return j

    augmented_pairs = np.array(
        [translate(i, j) if j > 0 else j for i, j in enumerate(augmented_pairs)]
    )
    augmented_struct = pairs_to_struct(augmented_pairs)

    return augmented_seq, augmented_struct


def augment_mutation(seq, struct, mutate_frac=0.05):
    # Pick mutation locations
    num_mutations = round(mutate_frac / 0.75 * len(seq))
    mutation_inds = np.random.choice(
        np.arange(len(seq)), size=num_mutations, replace=False
    )
    augmented_seq = list(seq)
    for i in mutation_inds:
        augmented_seq[i] = np.random.choice(["A", "U", "C", "G"])
    augmented_seq = "".join(augmented_seq)

    # Adjust structure, remove base pairs for unstable mutated base pairs
    augmented_pairs = struct_to_pairs(struct)
    stable_base_pairs = [
        ("A", "U"),
        ("U", "A"),  # Watson-Crick
        ("G", "C"),
        ("C", "G"),  # Watson-Crick
        ("G", "U"),
        ("U", "G"),
    ]  # Wobble
    for i in mutation_inds:
        j = augmented_pairs[i] - 1
        if j >= 0:
            if (augmented_seq[i], augmented_seq[j]) not in stable_base_pairs:
                augmented_pairs[i] = 0
                augmented_pairs[j] = 0
    augmented_struct = pairs_to_struct(augmented_pairs)

    return augmented_seq, augmented_struct


def augment_reverse_complement(seq, struct):
    augmented_seq = seq
    augmented_struct = struct

    # Reverse complement
    augmented_seq = (
        seq[::-1]
        .replace("A", "?")
        .replace("U", "A")
        .replace("?", "U")
        .replace("G", "?")
        .replace("C", "G")
        .replace("?", "C")
    )

    # Reverse structure
    pairs = struct_to_pairs(struct)
    augmented_pairs = pairs[::-1]
    augmented_pairs = np.array(
        [len(seq) - j + 1 if j > 0 else j for j in augmented_pairs]
    )
    augmented_struct = pairs_to_struct(augmented_pairs)

    return augmented_seq, augmented_struct


def augment(seq, struct, min_augments=2, max_augments=2):
    # Sample augmentations
    augments = [
        augment_deletion,
        augment_insertion,
        augment_translocation,
        augment_inversion,
        augment_mutation,
        augment_reverse_complement,
    ]
    max_augments = min(max_augments, len(augments))
    min_augments = min(min_augments, max_augments)
    n_augments = np.random.randint(min_augments, max_augments + 1)
    sampled_augments = np.random.choice(
        np.arange(len(augments)), size=n_augments, replace=False
    )

    # Apply augmentations
    for i in sampled_augments:
        seq, struct = augments[i](seq, struct)

    return seq, struct
