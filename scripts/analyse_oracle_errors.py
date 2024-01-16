import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import re

from src.utils import get_scores, get_scores_df

# bpRNA_RFAM_42242, 2072, 0.139535
# bpRNA_RFAM_43108, 2771, 0.194921
# bpRNA_RFAM_43136, 3456, 0.253254
# bpRNA_CRW_55337, 4216, 0.332103
rna = "bpRNA_RFAM_43136"

results_path = Path("resources/results/sequencewise")
oracle_scores = get_scores_df(
    results_path / "divide_oracle_1000_rnaf_sequencewise.csv"
).set_index("rna_name")
seq = oracle_scores.loc[rna, "seq"]
struct = oracle_scores.loc[rna, "struct"]
pred = oracle_scores.loc[rna, "pred"]


def rnafold_predict(seq):
    tstart = time.time()
    output = os.popen(f"echo {seq} | RNAfold").read()
    pred = output.split("\n")[1].split(" ")[0]
    ttot = time.time() - tstart

    return pred, None, None, ttot, 0.0


def oracle_get_cuts(struct):
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


def divide_predict(
    seq,
    max_length=1000,
    max_steps=None,
    cut_model=None,
    predict_fnc=rnafold_predict,
    struct="",
    cuts_path=None,
    rna_name="",
):
    tstart = time.time()

    if len(seq) <= max_length or max_steps == 0:
        pred, a, b, ttot, memory = predict_fnc(seq)
        subpreds = [(np.array([[0, len(seq) - 1]]), seq, struct, pred)]
        return pred, a, b, ttot, memory, subpreds

    # Get cuts
    cuts, outer = oracle_get_cuts(struct)

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
    preds = []
    outer_preds = []
    memories = []
    max_steps = max_steps - 1 if max_steps is not None else None
    all_subpreds = []
    for left_b, right_b in inner_bounds:
        subseq = seq[left_b:right_b]
        substruct = struct[left_b:right_b]
        assert substruct.count("(") == substruct.count(")")
        pred, _, _, _, memory, subpreds = divide_predict(
            subseq,
            max_length=max_length,
            max_steps=max_steps,
            cut_model=cut_model,
            predict_fnc=predict_fnc,
            struct=substruct,
            cuts_path=cuts_path,
            rna_name=rna_name,
        )
        preds.append(pred)
        memories.append(memory)
        for r, se, st, pr in subpreds:
            all_subpreds.append((r + left_b, se, st, pr))

    if outer_bounds:
        left_subseq = seq[outer_bounds[0][0] : outer_bounds[0][1]]
        right_subseq = seq[outer_bounds[1][0] : outer_bounds[1][1]]
        subseq = left_subseq + right_subseq
        left_substruct = struct[outer_bounds[0][0] : outer_bounds[0][1]]
        right_substruct = struct[outer_bounds[1][0] : outer_bounds[1][1]]
        substruct = left_substruct + right_substruct
        assert substruct.count("(") == substruct.count(")")
        pred, _, _, _, memory, subpreds = divide_predict(
            subseq,
            max_length=max_length,
            max_steps=max_steps,
            cut_model=cut_model,
            predict_fnc=predict_fnc,
            struct=substruct,
            cuts_path=cuts_path,
            rna_name=rna_name,
        )

        left_pred, right_pred = pred[: len(left_subseq)], pred[len(left_subseq) :]
        outer_preds = [left_pred, right_pred]
        memories.append(memory)
        sep = outer_bounds[0][1] - outer_bounds[0][0]
        for r, se, st, pr in subpreds:
            lefts = r[r[:, 1] < sep]
            middle = r[np.all([r[:, 0] < sep, r[:, 1] >= sep], axis=0)]
            rights = r[r[:, 0] >= sep]
            middle_left = (
                np.array([[middle[0, 0], sep - 1]])
                if middle.size > 0
                else np.array([[]])
            )
            middle_right = (
                np.array([[sep, middle[0, 1]]]) if middle.size > 0 else np.array([[]])
            )
            new_r = np.vstack(
                [
                    lefts + outer_bounds[0][0],
                    middle_left + outer_bounds[0][0],
                    middle_right + outer_bounds[1][0] - sep,
                    rights + outer_bounds[1][0] - sep,
                ]
            )
            all_subpreds.append((new_r, se, st, pr))

    # Patch sub predictions into global prediction
    global_pred = "".join(preds)
    if outer_bounds:
        global_pred = outer_preds[0] + global_pred + outer_preds[1]
    memory = max(memories)
    ttot = time.time() - tstart

    return global_pred, None, None, ttot, memory, all_subpreds


def check_results(subpreds, seq, struct):
    nuc_count = {i: 0 for i in range(len(seq))}
    for r, se, st, pr in subpreds:
        # Check subpred integrity
        assert len(se) == (r[:, 1] - r[:, 0] + 1).sum()
        assert len(st) == (r[:, 1] - r[:, 0] + 1).sum()
        assert len(pr) == (r[:, 1] - r[:, 0] + 1).sum()
        assert set(se).issubset({"A", "U", "C", "G"})
        assert set(pr).issubset({"(", ")", "."})
        assert st.count("(") == st.count(")")
        assert pr.count("(") == pr.count(")")
        assert "".join([seq[x : y + 1] for x, y in r]) == se
        assert "".join([struct[x : y + 1] for x, y in r]) == st
        assert rnafold_predict(se)[0] == pr
        for x, y in r:
            for i in range(x, y + 1):
                nuc_count[i] += 1
    assert all([count == 1 for count in nuc_count.values()])
    print("Integrity check passed.")


global_pred, _, _, ttot, memory, subpreds = divide_predict(seq, struct=struct)
assert global_pred == pred
check_results(subpreds, seq, struct)

## Compute final cuts
with open(r"resources/data_structures/test_sequencewise.dbn", "r") as f:
    txt = f.read()
lines = txt.strip().split("\n")
names, seqs, structs = lines[::3], lines[1::3], lines[2::3]
names = [n.split("#Name: ")[1] for n in names]
subcuts = []
for se, st in zip(seqs, structs):
    _, _, _, _, _, subs = divide_predict(
        se, struct=st, predict_fnc=lambda seq: ("." * len(seq), None, None, 0.0, 0.0)
    )
    subs = [x[0].tolist() for x in subs]
    subcuts.append(subs)
df = pd.DataFrame({"rna_name": names, "split": subcuts})
df.to_csv(r"resources/results/test_sequencewise_allsplits.csv", index=False)

## Analyse correlations
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils import get_scores_df

# results_path = Path("resources/results")
# sers_scores = [get_scores_df(results_path / "divide_oracle_1000_mx_sequencewise.csv").fscore,
#                get_scores_df(results_path / "mxfold2_sequencewise.csv").fscore,
#                get_scores_df(results_path / "divide_oracle_1000_lf_sequencewise.csv").fscore,
#                get_scores_df(results_path / "linearfold_sequencewise.csv").fscore,
#                get_scores_df(results_path / "divide_oracle_1000_rnaf_sequencewise.csv").fscore,
#                get_scores_df(results_path / "rnafold_sequencewise.csv").fscore,
#                ]
# sers_names = ['fscore_oracle_mxfold2',
#               'fscore_mxfold2',
#               'fscore_oracle_linearfold',
#               'fscore_linearfold',
#               'fscore_oracle_rnafold',
#               'fscore_rnafold',
#               ]
#
# oracle_scores = pd.concat(sers_scores, axis=1)
# oracle_scores.columns = sers_names
# oracle_scores['split'] = df.split
# oracle_scores['frag_lengths'] = df.split.apply(
#                                 lambda s: [sum([end - start + 1 for start, end in f]) for f in s])
# oracle_scores['min_frag_length'] = oracle_scores.frag_lengths.apply(min)
# oracle_scores['mean_frag_length'] = oracle_scores.frag_lengths.apply(lambda x: sum(x) / len(x))
# oracle_scores['frags_number'] = oracle_scores.frag_lengths.apply(len)
# oracle_scores['fscore_averaged'] = oracle_scores.iloc[:, :len(sers_scores)].mean(axis=1)
# oracle_scores.sort_values('fscore_averaged', inplace=True)
# oracle_scores = oracle_scores[oracle_scores.frags_number >= 2]
# oracle_scores.drop('fscore_averaged', axis=1, inplace=True)
# oracle_scores.to_csv(r'resources/results/oracle_sequencewise_analyse_splits.csv', index=False)


results_path = Path("resources/results/sequencewise")
sers_preds = [
    get_scores_df(results_path / "divide_oracle_1000_mx_sequencewise.csv")[
        ["struct", "pred"]
    ],
    get_scores_df(results_path / "divide_oracle_1000_lf_sequencewise.csv").pred,
    get_scores_df(results_path / "divide_oracle_1000_rnaf_sequencewise.csv").pred,
]
cols = [
    "struct",
    "oracle_mxfold2",
    "oracle_linearfold",
    "oracle_rnafold",
]

oracle_preds = pd.concat(sers_preds, axis=1)
oracle_preds.columns = cols
oracle_preds["split"] = df.split
seq_lengths = []
seq_fscores_mx = []
seq_fscores_lf = []
seq_fscores_rnaf = []
frag_lengths = []
frag_n_parts = []
frag_fscores_mx = []
frag_fscores_lf = []
frag_fscores_rnaf = []
for i in range(oracle_preds.shape[0]):
    struct, mxfold2, linearfold, rnafold, split = oracle_preds.iloc[i]
    seq_len = len(struct)
    _, _, seq_fscore_mx, _ = get_scores(struct, mxfold2)
    _, _, seq_fscore_lf, _ = get_scores(struct, linearfold)
    _, _, seq_fscore_rnaf, _ = get_scores(struct, rnafold)
    for frag in split:
        st = "".join([struct[start : end + 1] for start, end in frag])
        mx = "".join([mxfold2[start : end + 1] for start, end in frag])
        lf = "".join([linearfold[start : end + 1] for start, end in frag])
        rnaf = "".join([rnafold[start : end + 1] for start, end in frag])

        _, _, frag_fscore_mx, _ = get_scores(st, mx)
        _, _, frag_fscore_lf, _ = get_scores(st, lf)
        _, _, frag_fscore_rnaf, _ = get_scores(st, rnaf)

        seq_lengths.append(seq_len)
        seq_fscores_mx.append(seq_fscore_mx)
        seq_fscores_lf.append(seq_fscore_lf)
        seq_fscores_rnaf.append(seq_fscore_rnaf)
        frag_lengths.append(sum([end - start + 1 for start, end in frag]))
        frag_n_parts.append(len(frag))
        frag_fscores_mx.append(frag_fscore_mx)
        frag_fscores_lf.append(frag_fscore_lf)
        frag_fscores_rnaf.append(frag_fscore_rnaf)

df_frags = pd.DataFrame(
    {
        "seq_length": seq_lengths,
        "seq_fscore_oracle_mxfold2": seq_fscores_mx,
        "seq_fscore_oracle_linearfold": seq_fscores_lf,
        "seq_fscore_oracle_rnafold": seq_fscores_rnaf,
        "frag_length": frag_lengths,
        "frag_n_parts": frag_n_parts,
        "frag_fscore_mxfold2": frag_fscores_mx,
        "frag_fscore_linearfold": frag_fscores_lf,
        "frag_fscore_rnafold": frag_fscores_rnaf,
    }
)
df_frags.sort_values("frag_fscore_mxfold2", inplace=True)
df_frags.to_csv(r"resources/results/fragment_fscores.csv", index=False)
frag_bins = np.array(
    [df_frags.frag_length.min(), 50, 200, 400, 600, df_frags.frag_length.max() + 1]
)
df_frags["frag_bin"] = df_frags.frag_length.apply(
    lambda x: len(frag_bins) - np.argmax(x >= frag_bins[::-1]) - 1
)
fragtickslabels = [
    f"{str(a)}-{str(b-1)} nc.\n{int(df_frags[df_frags.frag_bin == i].shape[0])} fragments"
    for i, (a, b) in enumerate(zip(frag_bins[:-1], frag_bins[1:]))
]
seq_bins = np.array(
    [df_frags.seq_length.min(), 1000, 2000, 3000, 4000, df_frags.seq_length.max() + 1]
)
df_frags["seq_bin"] = df_frags.seq_length.apply(
    lambda x: len(seq_bins) - np.argmax(x >= seq_bins[::-1]) - 1
)
seqtickslabels = [
    f"{str(a)}-{str(b-1)} nc.\n{int(df_frags[df_frags.seq_bin == i].shape[0])} fragments"
    for i, (a, b) in enumerate(zip(seq_bins[:-1], seq_bins[1:]))
]

data_hue = pd.melt(
    df_frags,
    id_vars=[c for c in df_frags.columns if "frag_fscore" not in c],
    value_vars=[c for c in df_frags.columns if "frag_fscore" in c],
)
plt.figure()
ax = sns.boxplot(data_hue, x="frag_bin", y="value", hue="variable")
ax.set_xticklabels(seqtickslabels)
ax.set_xlabel("Length")
ax.set_ylabel("Fscore")
ax.set_title(f"Fscore vs fragment length")
plt.show()

plt.figure(figsize=(10, 7.5))
data_heat = pd.pivot_table(
    df_frags,
    values="frag_fscore_mxfold2",
    index="frag_bin",
    columns="seq_bin",
    aggfunc="mean",
)
data_heat = data_heat.loc[
    np.arange(len(frag_bins) - 1).tolist()[::-1], np.arange(len(seq_bins) - 1).tolist()
]
data_heat.index = [y.split("\n")[0] for y in fragtickslabels[::-1]]
data_heat.columns = [x.split("\n")[0] for x in seqtickslabels]
sns.heatmap(data_heat, cmap="hot", cbar_kws={"label": "F-score"})
plt.xlabel("Sequence length")
plt.ylabel("Fragment length")
plt.title("F-score depending on sequence length X fragment length")
for i in np.arange(len(seq_bins) - 1):
    for j in np.arange(len(frag_bins) - 1):
        n_frags = df_frags[(df_frags.seq_bin == i) & (df_frags.frag_bin == j)].shape[0]
        val = data_heat.iloc[len(frag_bins) - 2 - j, i]
        color = "white" if val < 0.4 else "black"
        plt.text(
            i + 0.5,
            len(frag_bins) - 2 - j + 0.5,
            f"{round(val, 2)}\n{n_frags} fragments",
            color=color,
            ha="center",
            va="center",
        )
plt.show()

# rnasubopt (rnafold / plusieurs structs)
# minimum threshold for subseq length ? (ou n découpages max ?)
#
# graphe avec chaque feuille = 2 noeud avec n aretes de valeur énergie
# => k plus court chemin pour k recombinaisons
#
# tableau avec preds + découpages finaux
# corrélation perf / nombre de découpages / taille minimale ou moyenne (calculer découpages finaux de l'oracle sur la base, croiser avec preds)
