import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from src.utils import get_scores_df

## Score predictions
# module load viennarna/2.5.0
# module load pytorch-gpu/py3/2.5.0
# module load git git-lfs
# python sampling.py --out_path output.seq --max_new_tokens 1024 --ckpt_path model_updated.pt --tokenizer_path tokenizer

# folder = "familywise"
# results_path = Path("resources/results/predictions/") / folder
# files = os.listdir(results_path)
#
#
# def filename_to_model_name(filename):
#     filename = filename.split(".csv")[0]
#     filename = (
#         filename.replace("_sequencewise", "")
#         .replace("_familywise", "")
#         .replace("_16S23S", "")
#     )
#     parts = filename.split("_")
#     model_transform = {
#         "dividefold": "DivideFold",
#         "linearfold": "LinearFold",
#         "mxfold2": "MXfold2",
#         "rnafold": "RNAfold",
#         "knotfold": "KnotFold",
#         "mx": "MXfold2",
#         "lf": "LinearFold",
#         "rnaf": "RNAfold",
#         "kf": "KnotFold",
#         "ipk": "IPknot",
#         "pbk": "ProbKnot",
#         "pks": "pKiss",
#         "ipknot": "IPknot",
#         "pkiss": "pKiss",
#         "probknot": "ProbKnot",
#         "ufold": "UFold",
#     }
#     if parts[0] != "dividefold":
#         model = model_transform[parts[0]]
#         return model
#
#     cut_model = (
#         parts[1]
#         .replace("cnn", "CNN")
#         .replace("mlp", "MLP")
#         .replace("bilstm", "BiLSTM")
#         .replace("oracle", "Oracle")
#     )
#     model = f"{model_transform[parts[0]]} {cut_model} ({parts[2]}) + {model_transform[parts[3]]}"
#
#     return model
#
#
# scores = [
#     get_scores_df(results_path / f)
#     .assign(model=filename_to_model_name(f))
#     for f in files
# ]
#
# data_scores = pd.concat(scores).reset_index(drop=True)
# assert data_scores.groupby("rna_name").struct.nunique().max() == 1
# data_scores.to_csv(rf"resources/results/{folder}.csv", index=False)
# data_scores[(data_scores.pk_motif_tp + data_scores.pk_motif_fn) > 0].to_csv(rf"resources/results/{folder}_pk.csv", index=False)
# data_scores[(data_scores.pk_motif_tp + data_scores.pk_motif_fn) == 0].to_csv(rf"resources/results/{folder}_nopk.csv", index=False)


def plot(
    data,
    yvar,
    bins=[400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000],
    hue="model",
    how="boxplot",
    ax_to_plot=None,
    **kwargs,
):
    data = data.copy()  # data[data.length >= 1000].copy()
    bins = np.array(bins)
    data = data[(data.length >= bins[0]) & (data.length < bins[-1])]
    data["bin"] = data.length.apply(
        lambda x: len(bins) - np.argmax(x >= bins[::-1]) - 1
    )
    ax = ax_to_plot
    if ax_to_plot is None:
        fig, ax = plt.subplots(figsize=(20, 6))
    ax.tick_params(axis="both", which="major", labelsize=11)

    plot_fnc = None
    if how == "boxplot":
        plot_fnc = sns.boxplot
    elif how == "barplot":
        plot_fnc = sns.barplot

    ref_palette = mcolors.TABLEAU_COLORS
    palette = {
        "DivideFold + KnotFold": ref_palette["tab:blue"],
        "IPknot": ref_palette["tab:orange"],
        "ProbKnot": ref_palette["tab:green"],
        "KnotFold": ref_palette["tab:red"],
        "pKiss": ref_palette["tab:purple"],
        "UFold": ref_palette["tab:brown"],
        "100 nt": ref_palette["tab:gray"],
        "200 nt": ref_palette["tab:blue"],
        "300 nt": ref_palette["tab:pink"],
        "400 nt": ref_palette["tab:orange"],
        "500 nt": ref_palette["tab:cyan"],
        "600 nt": ref_palette["tab:green"],
        "800 nt": ref_palette["tab:red"],
        "1000 nt": ref_palette["tab:purple"],
        "1200 nt": ref_palette["tab:brown"],
    }
    ax = plot_fnc(
        data,
        x="bin",
        y=yvar,
        hue=hue,
        hue_order=data[hue].unique(),
        ax=ax,
        palette=palette,
        **kwargs,
    )
    xtickslabels = [
        f"{str(a)}-{str(b-1)} nt\n{data[data.bin == i].rna_name.nunique()} RNAs"
        for i, (a, b) in enumerate(zip(bins[:-1], bins[1:]))
        if i in data.bin.unique()
    ]
    ax.set_xticklabels(xtickslabels)
    ax.set_xlabel("Sequence length", fontsize=16)
    if yvar == "time":
        ylabel = "Time (s)"
    elif yvar == "fscore":
        ylabel = "F-score"
    elif yvar == "pk_motif_sen":
        ylabel = "Recall"
    elif yvar == "pk_motif_fscore":
        ylabel = "F-score"
    ax.set_ylabel(ylabel, fontsize=16)
    if ("legend" not in kwargs) or (kwargs["legend"]):
        loc = "upper right" if yvar != "time" else "upper left"
        ax.legend(loc=loc, title=hue.capitalize(), prop={"size": 11}, title_fontsize=11)


## Pseudoknots table
def format_pk_table(df, min_len=1000):
    df = df.copy()
    df = df[
        (~df.model.apply(lambda x: "DivideFold" in x))
        | ((df.cut_compression > 0) & (df.length >= min_len))
    ]
    df = df[~df.model.str.contains("Oracle")]
    subdfs = []
    all_model_groups = [
        ("DivideFold", "KnotFold"),
        ("IPknot",),
        ("ProbKnot",),
        ("KnotFold",),
        ("pKiss",),
        ("UFold",),
    ]
    for model_groups in all_model_groups:
        conds = [df.model.str.contains(x) for x in model_groups]
        if len(conds) == 1:
            conds.append(~df.model.str.contains("DivideFold"))
        idx = pd.concat(conds, axis=1).all(axis=1)
        subdata = df[idx].sort_values(
            ["pk_motif_tp", "pk_motif_fscore"], ascending=False
        )
        subdf = subdata.groupby("rna_name").first().reset_index()
        # subdata = df[idx]
        # chosen_model = subdata.groupby("model")["pk_motif_sen"].mean() \
        #                                                        .sort_values(ascending=False) \
        #                                                        .index[0]
        # subdf = subdata[subdata.model == chosen_model].reset_index(drop=True)
        subdf.loc[:, "model"] = " + ".join(model_groups)
        subdfs.append(subdf)
    df = pd.concat(subdfs).reset_index(drop=True)
    return df


# Sequence-wise
df_pktable = pd.read_csv(rf"resources/results/sequencewise_pk.csv")
df_pktable = format_pk_table(df_pktable)
df_pktable = df_pktable[df_pktable.length >= 1000]  # RNAs longer than 1000 nt
pk_table = (
    df_pktable.groupby("model")[["pk_motif_sen", "pk_motif_ppv", "pk_motif_fscore"]]
    .mean()
    .sort_values("pk_motif_sen", ascending=False)
    .apply(lambda x: round(x, 3))
)
pk_table.columns = ["Recall", "Precision", "F-score"]

# Family-wise
df_pktable = pd.read_csv(rf"resources/results/familywise_pk.csv")
df_pktable = format_pk_table(df_pktable, min_len=500)
df_pktable = df_pktable[df_pktable.length >= 500]  # RNAs longer than 500 nt
pk_table = (
    df_pktable.groupby("model")[["pk_motif_sen", "pk_motif_ppv", "pk_motif_fscore"]]
    .mean()
    .sort_values("pk_motif_sen", ascending=False)
    .apply(lambda x: round(x, 3))
)
pk_table.columns = ["Recall", "Precision", "F-score"]

## Secondary structure table
# Sequence-wise
df_fscoretable = pd.read_csv(rf"resources/results/sequencewise_pk.csv")
df_fscoretable = format_pk_table(df_fscoretable)
df_fscoretable = df_fscoretable[~df_fscoretable.model.isin(["pKiss"])]
df_fscoretable = df_fscoretable[
    df_fscoretable.length >= 1000
]  # RNAs longer than 1000 nt
fscore_table = (
    df_fscoretable.groupby("model")[["sen", "ppv", "fscore"]]
    .mean()
    .sort_values("fscore", ascending=False)
    .apply(lambda x: round(x, 3))
)
fscore_table.columns = ["Recall", "Precision", "F-score"]

# Family-wise
df_fscoretable = pd.read_csv(rf"resources/results/familywise_pk.csv")
df_fscoretable = format_pk_table(df_fscoretable, min_len=500)
df_fscoretable = df_fscoretable[df_fscoretable.length >= 500]  # RNAs longer than 500 nt
fscore_table = (
    df_fscoretable.groupby("model")[["sen", "ppv", "fscore"]]
    .mean()
    .sort_values("fscore", ascending=False)
    .apply(lambda x: round(x, 3))
)
fscore_table.columns = ["Recall", "Precision", "F-score"]

## Pseudoknots table 16S 23S
df_pktable = pd.read_csv(rf"resources/results/16S23S_pk.csv")
df_pktable = format_pk_table(df_pktable)
df_pktable["n_pk_motif"] = df_pktable.pk_motif_tp + df_pktable.pk_motif_fn
table_16S = (
    df_pktable[df_pktable.rna_name == "CRW_16S_B_Ac_75"]
    .sort_values(["pk_motif_tp", "pk_motif_fscore"], ascending=False)[
        ["model", "pk_motif_tp", "n_pk_motif", "pk_motif_fp"]
    ]
    .set_index("model", drop=True)
)
table_23S = (
    df_pktable[df_pktable.rna_name == "CRW_23S_B_F_46"]
    .sort_values(["pk_motif_tp", "pk_motif_fscore"], ascending=False)[
        ["model", "pk_motif_tp", "n_pk_motif", "pk_motif_fp"]
    ]
    .set_index("model", drop=True)
)

## Secondary structure table 16S 23S
df_pktable = pd.read_csv(rf"resources/results/16S23S_pk.csv")
df_pktable = format_pk_table(df_pktable)
table_16S = (
    df_pktable[df_pktable.rna_name == "CRW_16S_B_Ac_75"]
    .sort_values("fscore", ascending=False)[["model", "sen", "ppv", "fscore", "time"]]
    .set_index("model", drop=True)
    .apply(lambda x: round(x, 3))
)
table_23S = (
    df_pktable[df_pktable.rna_name == "CRW_23S_B_F_46"]
    .sort_values("fscore", ascending=False)[["model", "sen", "ppv", "fscore", "time"]]
    .set_index("model", drop=True)
    .apply(lambda x: round(x, 3))
)

## Structure prediction models sequence-wise
data_scores = pd.read_csv(rf"resources/results/sequencewise_pk.csv")
data_scores = data_scores[
    (~data_scores.model.apply(lambda x: "DivideFold" in x))
    | (data_scores.cut_compression > 0)
]
data = pd.concat(
    [
        data_scores[data_scores.model == x]
        for x in [
            "IPknot",
            "ProbKnot",
            "KnotFold",
            "pKiss",
            "UFold",
        ]
    ]
)
assert np.all(data.groupby("rna_name").model.nunique() == data.model.nunique())

data.loc[(data.model == "UFold") & (data.length == 600), "length"] = 599
fig, axs = plt.subplots(2, sharex=True, figsize=(7, 6))
plot(data, "fscore", bins=[400, 600, 800, 1000], ax_to_plot=axs[0], legend=False)
axs[0].set_ylim(bottom=0.0, top=1.0)
plot(
    data,
    "time",
    bins=[400, 600, 800, 1000],
    ax_to_plot=axs[1],
    log_scale=True,
    legend=False,
)
axs[1].set_ylim(bottom=1, top=2000)
plt.show()

## Structure prediction models family-wise
data_scores = pd.read_csv(rf"resources/results/familywise_pk.csv")
data_scores = data_scores[
    (~data_scores.model.apply(lambda x: "DivideFold" in x))
    | (data_scores.cut_compression > 0)
]
data = pd.concat(
    [
        data_scores[data_scores.model == x]
        for x in [
            "IPknot",
            "ProbKnot",
            "KnotFold",
            "pKiss",
            "UFold",
        ]
    ]
)
assert np.all(data.groupby("rna_name").model.nunique() == data.model.nunique())

data.loc[(data.model == "UFold") & (data.length == 600), "length"] = 599
fig, axs = plt.subplots(2, sharex=True, figsize=(7, 6))
plot(data, "fscore", bins=[400, 600, 800, 1000], ax_to_plot=axs[0], legend=False)
axs[0].set_ylim(bottom=0.0, top=1.0)
plot(
    data,
    "time",
    bins=[400, 600, 800, 1000],
    ax_to_plot=axs[1],
    log_scale=True,
    legend=False,
)
axs[1].set_ylim(bottom=1, top=2000)
plt.show()

## Hyperparameter evaluation sequence-wise
data_scores = pd.read_csv(rf"resources/results/sequencewise_pk.csv")
data_scores = data_scores[
    (~data_scores.model.apply(lambda x: "DivideFold" in x))
    | (data_scores.cut_compression > 0)
]
data = pd.concat(
    [
        data_scores[data_scores.model == x]
        for x in [
            "DivideFold CNN400 (200) + KnotFold",
            "DivideFold CNN400 (400) + KnotFold",
            "DivideFold CNN400 (600) + KnotFold",
            "DivideFold CNN400 (800) + KnotFold",
            "DivideFold CNN400 (1000) + KnotFold",
            "DivideFold CNN400 (1200) + KnotFold",
        ]
    ]
)
data["Maximum fragment length"] = data.model.apply(
    lambda x: x.split("(")[1].split(")")[0] + " nt"
)
assert np.all(data.groupby("rna_name").model.nunique() == data.model.nunique())

plot(
    data,
    "fscore",
    hue="Maximum fragment length",
    bins=[1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000],
)
plt.ylim(bottom=0.1, top=0.9)
plt.show()

## Hyperparameter evaluation family-wise
data_scores = pd.read_csv(rf"resources/results/familywise_pk.csv")
data_scores = data_scores[
    (~data_scores.model.apply(lambda x: "DivideFold" in x))
    | (data_scores.cut_compression > 0)
]
data = pd.concat(
    [
        data_scores[data_scores.model == x]
        for x in [
            "DivideFold CNN400 (100) + IPknot",
            "DivideFold CNN400 (200) + IPknot",
            "DivideFold CNN400 (300) + IPknot",
            "DivideFold CNN400 (400) + IPknot",
            "DivideFold CNN400 (500) + IPknot",
        ]
    ]
)
data["Maximum fragment length"] = data.model.apply(
    lambda x: x.split("(")[1].split(")")[0] + " nt"
)
assert np.all(data.groupby("rna_name").model.nunique() == data.model.nunique())

plot(
    data,
    "fscore",
    hue="Maximum fragment length",
    bins=[500, 600, 700, 800, 900, 1000],
)
plt.ylim(bottom=0.1, top=0.9)
plt.show()

## Pseudoknot graph sequence-wise
data_scores = pd.read_csv(rf"resources/results/sequencewise_pk.csv")
data = format_pk_table(data_scores)
assert np.all(data.groupby("rna_name").model.nunique() == data.model.nunique())

data = data[~data.model.isin(["pKiss", "UFold"])]
fig, axs = plt.subplots(2, sharex=True, figsize=(20, 10))
plot(
    data,
    "pk_motif_sen",
    bins=[1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000],
    how="barplot",
    ax_to_plot=axs[0],
)
axs[0].set_ylim(bottom=0.0, top=1.0)
plot(
    data,
    "pk_motif_fscore",
    bins=[1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000],
    how="barplot",
    ax_to_plot=axs[1],
)
axs[1].set_ylim(bottom=0.0, top=0.15)
plt.show()

## Secondary structure graph with pk sequence-wise
data_scores = pd.read_csv(rf"resources/results/sequencewise_pk.csv")
data_scores = data_scores[
    (~data_scores.model.apply(lambda x: "DivideFold" in x))
    | (data_scores.cut_compression > 0)
]
data = pd.concat(
    [
        data_scores[data_scores.model == x]
        for x in [
            "DivideFold CNN1600EVOAUGINCRANGE (1000) + KnotFold",
            "IPknot",
            "ProbKnot",
            "KnotFold",
        ]
    ]
)
assert np.all(data.groupby("rna_name").model.nunique() == data.model.nunique())

data.model = data.model.apply(lambda x: x.replace("CNN1600EVOAUGINCRANGE (1000) ", ""))

plot(data, "fscore", bins=[1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000])
plt.ylim([0, 1])
plot(
    data,
    "time",
    bins=[1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000],
    log_scale=True,
)
plt.ylim([3, 10000])
plt.show()

## Pseudoknot graph family-wise
data_scores = pd.read_csv(rf"resources/results/familywise_pk.csv")
data_scores.loc[
    (data_scores.model == "UFold") & (data_scores.length == 600), "length"
] = 599
data = format_pk_table(data_scores, min_len=500)
assert np.all(data.groupby("rna_name").model.nunique() == data.model.nunique())

fig, axs = plt.subplots(2, sharex=True, figsize=(20, 10))
plot(
    data,
    "pk_motif_sen",
    bins=[500, 600, 700, 800, 900, 1000],
    how="barplot",
    ax_to_plot=axs[0],
)
axs[0].set_ylim(bottom=0.0, top=1.0)
plot(
    data,
    "pk_motif_fscore",
    bins=[500, 600, 700, 800, 900, 1000],
    how="barplot",
    ax_to_plot=axs[1],
)
axs[1].set_ylim(bottom=0.0, top=0.15)
plt.show()

## Secondary structure graph with pk family-wise
data_scores = pd.read_csv(rf"resources/results/familywise_pk.csv")
data_scores = data_scores[
    (~data_scores.model.apply(lambda x: "DivideFold" in x))
    | (data_scores.cut_compression > 0)
]
data = pd.concat(
    [
        data_scores[data_scores.model == x]
        for x in [
            "DivideFold CNN1600EVOAUGINCRANGE (500) + KnotFold",
            "IPknot",
            "ProbKnot",
            "KnotFold",
            "pKiss",
            "UFold",
        ]
    ]
)
assert np.all(data.groupby("rna_name").model.nunique() == data.model.nunique())

data.model = data.model.apply(lambda x: x.replace("CNN1600EVOAUGINCRANGE (500) ", ""))
data.loc[(data.model == "UFold") & (data.length == 600), "length"] = 599

plot(data, "fscore", bins=[500, 600, 700, 800, 900, 1000])
plt.ylim([0, 1])
plot(
    data,
    "time",
    bins=[500, 600, 700, 800, 900, 1000],
    log_scale=True,
)
plt.ylim([3, 10000])
plt.show()

## Pseudoknot graph with pk sequence-wise on 500-1000 nt
data_scores = pd.read_csv(rf"resources/results/sequencewise_pk.csv")
data = format_pk_table(data_scores, min_len=500)
assert np.all(data.groupby("rna_name").model.nunique() == data.model.nunique())

data.loc[(data.model == "UFold") & (data.length == 600), "length"] = 599
fig, axs = plt.subplots(2, sharex=True, figsize=(20, 10))
plot(
    data,
    "pk_motif_sen",
    bins=[500, 600, 700, 800, 900, 1000],
    how="barplot",
    ax_to_plot=axs[0],
)
axs[0].set_ylim(bottom=0.0, top=1.0)
plot(
    data,
    "pk_motif_fscore",
    bins=[500, 600, 700, 800, 900, 1000],
    how="barplot",
    ax_to_plot=axs[1],
)
axs[1].set_ylim(bottom=0.0, top=0.2)
plt.show()

## Secondary structure graph with pk sequence-wise on 500-1000 nt
data_scores = pd.read_csv(rf"resources/results/sequencewise_pk.csv")
data_scores = data_scores[
    (~data_scores.model.apply(lambda x: "DivideFold" in x))
    | (data_scores.cut_compression > 0)
]
data = pd.concat(
    [
        data_scores[data_scores.model == x]
        for x in [
            "DivideFold CNN400 (400) + KnotFold",
            "IPknot",
            "ProbKnot",
            "KnotFold",
            "pKiss",
            "UFold",
        ]
    ]
)
assert np.all(data.groupby("rna_name").model.nunique() == data.model.nunique())

data.model = data.model.apply(lambda x: x.replace("CNN400 (400) ", ""))
data.loc[(data.model == "UFold") & (data.length == 600), "length"] = 599

plot(data, "fscore", bins=[500, 600, 700, 800, 900, 1000], legend=True)
plt.ylim([0, 1])
plot(
    data,
    "time",
    bins=[500, 600, 700, 800, 900, 1000],
    log_scale=True,
)
plt.ylim([3, 10000])
plt.show()
