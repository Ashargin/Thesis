import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_scores_df

## Score predictions
results_path = Path("resources/results/predictions/sequencewise")
files = os.listdir(results_path)


def filename_to_model_name(filename):
    filename = filename.split(".csv")[0]
    filename = (
        filename.replace("_sequencewise", "")
        .replace("_familywise", "")
        .replace("_16S23S", "")
    )
    parts = filename.split("_")
    model_transform = {
        "dividefold": "DivideFold",
        "linearfold": "LinearFold",
        "mxfold2": "MXfold2",
        "rnafold": "RNAfold",
        "mx": "MXfold2",
        "lf": "LinearFold",
        "rnaf": "RNAfold",
    }
    if parts[0] != "dividefold":
        model = model_transform[parts[0]]
        return model

    cut_model = (
        parts[1]
        .replace("cnn", "CNN")
        .replace("mlp", "MLP")
        .replace("bilstm", "BiLSTM")
        .replace("oracle", "Oracle")
    )
    model = f"{model_transform[parts[0]]} {cut_model} ({parts[2]}) + {model_transform[parts[3]]}"

    return model


scores = [
    get_scores_df(results_path / f).assign(model=filename_to_model_name(f))
    for f in files
]

data_scores = pd.concat(scores).reset_index(drop=True)


def plot(data, yvar):
    data = data.copy()  # data[data.length >= 1000].copy()
    bins = np.array(
        [400, 600, 800, 1000, 1200, 1400, 1700, 2700, 3000, 3400, 3800, 4400]
    )
    data = data[data.length >= bins[0]]
    data["bin"] = data.length.apply(
        lambda x: len(bins) - np.argmax(x >= bins[::-1]) - 1
    )
    plt.figure()
    ax = sns.boxplot(data, x="bin", y=yvar, hue="model")  # , palette="tab20")
    xtickslabels = [
        f"{str(a)}-{str(b-1)} nc.\n{data[data.bin == i].rna_name.nunique()} RNAs"
        for i, (a, b) in enumerate(zip(bins[:-1], bins[1:]))
        if i in data.bin.unique()
    ]
    ax.set_xticklabels(xtickslabels)
    ax.set_xlabel("Length")
    ax.set_ylabel(yvar.capitalize())
    # plt.title(f"{yvar.capitalize()} vs length")
    ax.legend(loc="upper right")


def plot_all(data):
    plot(data, "fscore")
    plt.ylim([0, 1])
    # plot(data, "mcc")
    plot(data, "time")
    plt.ylim([0, 200])
    plot(data, "memory")
    if data.cut_break_rate.isna().sum() == 0:
        plot(data, "cut_break_rate")
        plt.ylim([0, 1])
        plot(data, "cut_compression")
    plt.show()


# Sequencewise
data = pd.concat(
    [
        data_scores[data_scores.model == x]
        for x in [
            # "RNAfold",
            # "LinearFold",
            # "MXfold2",
            # "DivideFold MLP (meta) + RNAfold",
            # "DivideFold MLP (meta) + LinearFold",
            # "DivideFold MLP (meta) + MXfold2",
            # "DivideFold BiLSTM (meta) + RNAfold",
            # "DivideFold BiLSTM (meta) + LinearFold",
            # "DivideFold BiLSTM (meta) + MXfold2",
            # "DivideFold CNN (200) + RNAfold",
            # "DivideFold CNN (200) + LinearFold",
            # "DivideFold CNN (200) + MXfold2",
            # "DivideFold CNN (400) + RNAfold",
            # "DivideFold CNN (400) + LinearFold",
            # "DivideFold CNN (400) + MXfold2",
            # "DivideFold CNN (600) + RNAfold",
            # "DivideFold CNN (600) + LinearFold",
            # "DivideFold CNN (600) + MXfold2",
            # "DivideFold CNN (800) + RNAfold",
            # "DivideFold CNN (800) + LinearFold",
            # "DivideFold CNN (800) + MXfold2",
            # "DivideFold CNN (1000) + RNAfold",
            # "DivideFold CNN (1000) + LinearFold",
            # "DivideFold CNN (1000) + MXfold2",
            # "DivideFold CNN (1200) + RNAfold",
            # "DivideFold CNN (1200) + LinearFold",
            # "DivideFold CNN (1200) + MXfold2",
            "DivideFold CNN (meta) + RNAfold",
            "DivideFold CNN (meta) + LinearFold",
            "DivideFold CNN (meta) + MXfold2",
            # "DivideFold Oracle (200) + RNAfold",
            # DivideFold Oracle (200) + LinearFold",
            # "DivideFold Oracle (200) + MXfold2",
        ]
    ]
)


def clean_data(data):
    data = data.copy()
    if "fscore" in data.columns:
        data = data[data.fscore.notna()]
    # data = data[(data.length < 1650) | ((data.length > 2750))]
    seqs_all_models = data.groupby("seq").model.nunique() == data.model.nunique()
    # data = data[(data.seq.isin(seqs_all_models[seqs_all_models].index))]
    return data


data.model = data.model.apply(lambda x: x.replace(" (meta)", "").replace(" (200)", ""))
plot_all(clean_data(data))
plt.show()


# def get_ranking(data, start, end, reorder=True):
#     data = data.copy()
#     data_view = data[(data.length >= start) & (data.length < end)]
#     print(f"{start} to {end}: {data_view.rna_name.nunique()} RNAs")
#     df = pd.DataFrame(
#         {
#             "cut_break_rate": data_view.groupby("model").cut_break_rate.mean(),
#             "cut_compression": data_view.groupby("model").cut_compression.mean(),
#             "fscore": data_view.groupby("model").fscore.mean(),
#             "time": data_view.groupby("model").time.mean(),
#         }
#     )
#     if df.shape[0] > 0:
#         df = df.loc[data.model.unique(), :]
#     if reorder:
#         df = df.sort_values("fscore", ascending=False)
#     df.cut_break_rate = df.cut_break_rate.apply(
#         lambda x: np.nan if pd.isna(x) else str(round(100 * x, 2)) + "%"
#     )
#     df.cut_compression = df.cut_compression.apply(
#         lambda x: np.nan if pd.isna(x) else str(round(100 * x)) + "%"
#     )
#     df.fscore = df.fscore.apply(lambda x: round(x, 4))
#     return df
#
#
# ranks = get_ranking(data, 1000, 5000)
# ranks.to_excel(r"resources/results/cutting_metrics.xlsx")
# 200;400: DO NOT CUT
# 400;1300: 1 step
# 1300;1600: 200
# 1600;2000: 1 step
# 2000;4300: 2000
# ranks = [
#     get_ranking(data, i * 100, (i + 1) * 100, reorder=False)[["fscore"]].rename(
#         columns={"fscore": f"fscore_{i*100}-{(i+1)*100}"}
#     )
#     for i in range(43)
# ]
# ranks = [r for r in ranks if r.shape[0]]
# ranks = pd.concat(ranks, axis=1)
#
# X = [int(c.split("_")[1].split("-")[0]) for c in ranks.columns]
# ranks = ranks.loc[[idx for idx in ranks.index if "fusion" not in idx], :]
# cmap = plt.colormaps["jet"]
# for i, idx in enumerate(ranks.index):
#     color = cmap(i / ranks.shape[0])  # if "steps" not in idx else "black"
#     plt.plot(X, ranks.loc[idx, :], color=color, label=idx)
# plt.legend(loc="lower left", prop={"size": 7})
# plt.show()
