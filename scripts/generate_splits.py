import os
import time
import re
import numpy as np
import pandas as pd
import itertools
from pathlib import Path

from src.utils import optimize_pseudoknots, struct_to_pairs, pairs_to_struct


def oracle_get_cuts(struct):
    if len(struct) <= 3:
        return [], True

    # Determine depth levels
    struct = re.sub("[^\(\)\.]", ".", struct)
    depths = []
    count = 0
    for c in struct:
        if c == "(":
            depths.append(count)
            count += 1
        elif c == ")":
            depths.append(count - 1)
            count -= 1
        else:
            depths.append(-1)
    depths = np.array(depths)

    # Determine sequence cuts
    cuts = []
    d = -1
    for d in range(max(depths) + 1):
        if np.count_nonzero(depths == d) == 2:
            continue

        bounds = np.where(depths == d)[0]
        if d > 0:
            outer_bounds = np.where(depths == d - 1)[0]
            bounds = np.array([outer_bounds[0]] + list(bounds) + [outer_bounds[1]])
        else:
            bounds = bounds[1:-1]
        cuts = [
            int(np.ceil((bounds[i] + bounds[i + 1]) / 2))
            for i in np.arange(len(bounds))[::2]
        ]

        break

    # Edge cases
    if not cuts:
        if max(depths) == -1:  # no pairs
            cuts = [int(len(struct) / 2)]
        else:  # only stacking concentric pairs
            gaps = np.array(
                [
                    len(depths)
                    - np.argmax(depths[::-1] == d)
                    - 1
                    - np.argmax(depths == d)
                    for d in range(max(depths) + 1)
                ]
            )
            too_small = gaps <= len(struct) / 2
            if np.any(too_small):
                d = np.argmax(too_small)
                bounds = np.where(depths == d)[0]
                outer_bounds = (
                    np.where(depths == d - 1)[0]
                    if d > 0
                    else np.array([0, len(struct)])
                )
                outer_gap = outer_bounds[1] - outer_bounds[0]
                lbda = (len(struct) / 2 - gaps[d]) / (outer_gap - gaps[d])
                cuts = [
                    int(np.ceil(x + lbda * (y - x)))
                    for x, y in zip(bounds, outer_bounds)
                ]
                cuts[1] = max(cuts[1], bounds[1] + 1)
            else:
                d = max(depths)
                bounds = np.where(depths == d)[0]
                margin = gaps[-1] - len(struct) / 2
                cuts = [
                    int(np.ceil(bounds[0] + margin / 2)),
                    int(np.ceil(bounds[1] - margin / 2)),
                ]
                d += 1  # we force entering an artificial additional depth level

    if cuts[0] == 0:
        cuts = cuts[1:]
    if cuts[-1] == len(struct):
        cuts = cuts[:-1]
    assert cuts

    outer = d > 0

    return cuts, outer


def dividefold_get_fragment_ranges_preds(
    seq,
    rna_name,
    spans,
    max_length=50,
    max_steps=None,
    min_steps=0,
    cut_model=None,
    predict_fnc=None,
    max_motifs=None,
    fuse_to=None,
    struct="",
    return_cuts=True,
):
    tstart = time.time()

    if max_steps == 0 or len(seq) <= max_length and min_steps <= 0:
        pred, ttot, memory = (
            predict_fnc(seq) if not return_cuts else ("." * len(seq), 0.0, 0.0)
        )
        frag_preds = [(np.array([[0, len(seq) - 1]]).astype(int), pred)]
        return frag_preds, ttot, memory

    if struct:
        cuts, outer = oracle_get_cuts(struct)
        all_rna_names.append(rna_name)
        all_seqs.append(seq)
        all_structs.append(struct)
        all_cuts.append(str(cuts).replace(",", ""))
        all_outers.append(outer)
        all_spans.append(str(spans).replace(",", ""))
    else:
        cuts, outer = dividefold_get_cuts(
            seq, cut_model=cut_model, max_motifs=max_motifs, fuse_to=fuse_to
        )

    # Cut sequence into subsequences
    random_cuts = [int(len(seq) / 3), int(len(seq) * 2 / 3)]
    if not cuts:
        cuts = random_cuts
    if cuts[0] > 0:
        cuts = [0] + cuts
    if cuts[-1] < len(seq):
        cuts = cuts + [len(seq)]
    if len(cuts) < (4 if outer else 3):
        cuts = [0] + random_cuts + [len(seq)]
    assert np.all(np.array(cuts)[1:] > np.array(cuts)[:-1])

    outer_bounds = []
    inner_bounds = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    if outer:
        outer_bounds = [inner_bounds[0], inner_bounds[-1]]
        inner_bounds = inner_bounds[1:-1]

    # Predict subsequences
    frag_preds = []
    memories = []
    max_steps = max_steps - 1 if max_steps is not None else None
    min_steps -= 1
    for left_b, right_b in inner_bounds:
        subseq = seq[left_b:right_b]

        substruct = ""
        span_nucs = list(
            itertools.chain.from_iterable([range(a, b + 1) for a, b in spans])
        )
        span_nucs = np.array(span_nucs[left_b:right_b])
        cut_indices = np.argwhere(span_nucs[1:] != span_nucs[:-1] + 1).ravel() + 1
        cut_indices = np.append(np.append(0, cut_indices), len(span_nucs))
        new_spans = [
            (span_nucs[i], span_nucs[j - 1])
            for i, j in zip(cut_indices[:-1], cut_indices[1:])
        ]
        if struct:
            pairs = struct_to_pairs(struct)
            for i in range(left_b, right_b):
                j = pairs[i] - 1
                if j >= 0:
                    if (j < left_b) or (j >= right_b):
                        pairs[i] = 0
                    else:
                        pairs[i] -= left_b
            pairs = pairs[left_b:right_b]
            substruct = pairs_to_struct(pairs)
            assert substruct.count("(") == substruct.count(")")
        this_frag_preds, _, memory = dividefold_get_fragment_ranges_preds(
            subseq,
            rna_name,
            new_spans,
            max_length=max_length,
            max_steps=max_steps,
            min_steps=min_steps,
            cut_model=cut_model,
            predict_fnc=predict_fnc,
            max_motifs=max_motifs,
            fuse_to=fuse_to,
            struct=substruct,
            return_cuts=return_cuts,
        )

        for _range, pred in this_frag_preds:
            frag_preds.append((_range + left_b, pred))
        memories.append(memory)

    if outer_bounds:
        left_b_1, right_b_1 = outer_bounds[0]
        left_b_2, right_b_2 = outer_bounds[1]
        left_subseq = seq[left_b_1:right_b_1]
        right_subseq = seq[left_b_2:right_b_2]
        subseq = left_subseq + right_subseq

        substruct = ""
        span_nucs = list(
            itertools.chain.from_iterable([range(a, b + 1) for a, b in spans])
        )
        span_nucs = np.append(
            np.array(span_nucs[left_b_1:right_b_1]),
            np.array(span_nucs[left_b_2:right_b_2]),
        )
        cut_indices = np.argwhere(span_nucs[1:] != span_nucs[:-1] + 1).ravel() + 1
        cut_indices = np.append(np.append(0, cut_indices), len(span_nucs))
        new_spans = [
            (span_nucs[i], span_nucs[j - 1])
            for i, j in zip(cut_indices[:-1], cut_indices[1:])
        ]
        if struct:
            pairs = struct_to_pairs(struct)
            for i in range(left_b_1, right_b_1):
                j = pairs[i] - 1
                if j >= 0:
                    if ((j < left_b_1) or (j >= right_b_1)) and (
                        (j < left_b_2) or (j >= right_b_2)
                    ):
                        pairs[i] = 0
                    elif (j >= left_b_1) and (j < right_b_1):
                        pairs[i] -= left_b_1
                    else:
                        pairs[i] += right_b_1 - left_b_1 - left_b_2
            for i in range(left_b_2, right_b_2):
                j = pairs[i] - 1
                if j >= 0:
                    if ((j < left_b_1) or (j >= right_b_1)) and (
                        (j < left_b_2) or (j >= right_b_2)
                    ):
                        pairs[i] = 0
                    elif (j >= left_b_1) and (j < right_b_1):
                        pairs[i] -= left_b_1
                    else:
                        pairs[i] += right_b_1 - left_b_1 - left_b_2
            pairs = np.concatenate(
                [pairs[left_b_1:right_b_1], pairs[left_b_2:right_b_2]]
            ).astype(int)
            substruct = pairs_to_struct(pairs)
            assert substruct.count("(") == substruct.count(")")
        this_frag_preds, _, memory = dividefold_get_fragment_ranges_preds(
            subseq,
            rna_name,
            new_spans,
            max_length=max_length,
            max_steps=max_steps,
            min_steps=min_steps,
            cut_model=cut_model,
            predict_fnc=predict_fnc,
            max_motifs=max_motifs,
            fuse_to=fuse_to,
            struct=substruct,
            return_cuts=return_cuts,
        )

        sep = right_b_1 - left_b_1
        for _range, pred in this_frag_preds:
            lefts = _range[_range[:, 1] < sep]
            middle = _range[np.all([_range[:, 0] < sep, _range[:, 1] >= sep], axis=0)]
            rights = _range[_range[:, 0] >= sep]
            middle_left = (
                np.array([[middle[0, 0], sep - 1]])
                if middle.size > 0
                else np.zeros((0, 2))
            )
            middle_right = (
                np.array([[sep, middle[0, 1]]]) if middle.size > 0 else np.zeros((0, 2))
            )
            new_range = np.vstack(
                [
                    lefts + left_b_1,
                    middle_left + left_b_1,
                    middle_right + left_b_2 - sep,
                    rights + left_b_2 - sep,
                ]
            )
            frag_preds.append((new_range.astype(int), pred))
        memories.append(memory)

    # Patch sub predictions into global prediction
    memory = max(memories)
    ttot = time.time() - tstart

    return frag_preds, ttot, memory


def dividefold_predict(
    seq,
    rna_name,
    max_length=50,
    max_steps=None,
    min_steps=0,
    multipred_kmax=20,
    cut_model=None,
    predict_fnc=None,
    max_motifs=None,
    fuse_to=None,
    struct="",
    struct_to_print_fscores="",
    return_cuts=True,
):
    tstart = time.time()

    if max_length is None:
        if (predict_fnc is None) or (predict_fnc.__name__ != "knotfold_predict"):
            max_length = 2000 if len(seq) > 2500 else 400
        else:
            max_length = 1000

    if max_steps is not None and max_steps < min_steps:
        raise ValueError("max_steps must be greater than min_steps.")

    if struct:
        struct = optimize_pseudoknots(struct)
    if struct_to_print_fscores:
        struct_to_print_fscores = optimize_pseudoknots(struct_to_print_fscores)

    frag_preds, _, memory = dividefold_get_fragment_ranges_preds(
        seq,
        rna_name,
        [(1, len(seq))],
        max_length=max_length,
        max_steps=max_steps,
        min_steps=min_steps,
        cut_model=cut_model,
        predict_fnc=predict_fnc,
        max_motifs=max_motifs,
        fuse_to=fuse_to,
        struct=struct,
        return_cuts=return_cuts,
    )

    if return_cuts:
        ttot = time.time() - tstart
        return frag_preds, ttot, memory


path_structures = Path("resources/data_structures")
path_splits = Path("resources/data_splits")
split_dfs = []
for f in os.listdir(path_structures):
    print(f"Processing {f}...")
    global all_rna_names
    global all_seqs
    global all_structs
    global all_cuts
    global all_outers
    global all_spans
    all_rna_names = []
    all_seqs = []
    all_structs = []
    all_cuts = []
    all_outers = []
    all_spans = []
    txt = open(path_structures / f, "r").read()
    lines = txt.strip().split("\n")
    names, seqs, structs = lines[::3], lines[1::3], lines[2::3]
    for n, se, st in zip(names, seqs, structs):
        n = n.split("#Name: ")[1]
        st = optimize_pseudoknots(st)
        x = dividefold_predict(se, n, struct=st)

    split_df = pd.DataFrame(
        {
            "rna_name": all_rna_names,
            "span": all_spans,
            "seq": all_seqs,
            "struct": all_structs,
            "cuts": all_cuts,
            "outer": all_outers,
        }
    )
    split_df.to_csv(path_splits / f.replace(".dbn", ".csv"))
    split_dfs.append(split_df)

# for _, row in splits_df.iterrows():
#     ref_seq = main_df[main_df.rna_name == row.rna_name].iloc[0].seq
#     spans = [x.split("(")[-1] for x in row.span.split(")")[:-1]]
#     spans = [tuple(int(x) for x in sp.split(" ")) for sp in spans]
#     rebuilt = "".join([ref_seq[x - 1 : y] for x, y in spans])
#     assert rebuilt == row.seq
