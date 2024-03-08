import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_scores_df

## Score predictions
results_path = Path("resources/results/sequencewise")
files = os.listdir(results_path)


def filename_to_model_name(filename):
    if "dil" in filename and "INV" not in filename:
        raise Warning("Vanilla dilation should not be listed")

    filename = filename.split(".csv")[0]
    suffix = ""
    if "familywise" in filename:
        suffix = f' ({filename.split("_")[-1]}% similarity)'
        filename = "_".join(filename.split("_")[:-1])
    filename = (
        filename.replace("_sequencewise", "")
        .replace("_familywise", "")
        .replace("_16S23S", "")
        .replace("_cuttingmetrics", "")
    )
    parts = filename.split("_")
    model_transform = {
        "divide": "DivideFold",
        "linearfold": "LinearFold",
        "mxfold2": "MXfold2",
        "probknot": "ProbKnot",
        "rnafold": "RNAfold",
        "mx": "MXfold2",
        "lf": "LinearFold",
        "rnaf": "RNAfold",
        "ens": "Ensemble",
        "sub": "RNAsubopt",
    }
    if parts[0] != "divide":
        model = model_transform[parts[0]]
    else:
        stop_parts = re.findall(r"[^\W\d_]+|\d+", parts[2])
        stop_parts = [stop_parts[-1] + " nc."] + [
            f"{n} {s}" for n, s in zip(stop_parts[0::2], stop_parts[1::2])
        ]
        stop = (
            "(until "
            + ", ".join(stop_parts)
            .replace("mins", "min steps")
            .replace("meta nc.", "varying threshold")
            + ")"
        )
        model_parts = re.findall(r"[^\W\d_]+|\d+", parts[1])
        model_parts = [model_parts[0].upper()] + [
            f"{n} {s}" for n, s in zip(model_parts[1::2], model_parts[2::2])
        ]
        cut_model = (
            ", ".join(model_parts)
            .replace("ORACLE", "Oracle")
            .replace("dil", "dilation")
            .replace("INV", "")
            .replace("BILSTM", "BiLSTM")
            .replace("augmented", " augmented")
            .replace("fuse", "fusion threshold")
        )
        model = f"{model_transform[parts[0]]} {cut_model} {stop}"
        if len(parts) > 3:
            model = model + f" + {model_transform[parts[3]]}"
    model = model + suffix

    return model


scores = [
    get_scores_df(results_path / f).assign(model=filename_to_model_name(f))
    for f in files
]

data_scores = pd.concat(scores).reset_index(drop=True)


def plot(data, yvar):
    data = data[data.length >= 1000].copy()
    bins = np.array(
        [200, 400, 600, 800, 1000, 1200, 1400, 1700, 2700, 3000, 3400, 3800, 4400]
    )
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
    plt.title(f"{yvar.capitalize()} vs length")
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
            "RNAfold",
            "LinearFold",
            "MXfold2",
            "ProbKnot",
            "DivideFold BiLSTM, 50 motifs (until 1000 nc.) + MXfold2",
            "DivideFold BiLSTM, 100 motifs (until 1000 nc.) + MXfold2",
            "DivideFold BiLSTM, 200 motifs (until 1000 nc.) + MXfold2",
            "DivideFold MLP (until 1000 nc.) + LinearFold",
            "DivideFold MLP (until 1000 nc.) + RNAfold",
            "DivideFold MLP (until 1000 nc.) + RNAsubopt",
            "DivideFold MLP (until 1000 nc.) + Ensemble",
            "DivideFold MLP (until 1000 nc.) + MXfold2",
            "DivideFold CNN (until 1000 nc.) + LinearFold",
            "DivideFold CNN (until 1000 nc.) + RNAfold",
            "DivideFold CNN (until 1000 nc.) + RNAsubopt",
            "DivideFold CNN (until 1000 nc.) + Ensemble",
            "DivideFold CNN (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 0 motifs (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 10 motifs (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 25 motifs (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 1 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 2 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 4 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 8 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 16 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 32 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 64 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 128 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 256 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation (until 100 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation (until 200 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation (until 400 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation (until 600 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation (until 800 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation (until varying threshold) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 100 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 200 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 250 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 300 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 350 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 400 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 400 fusion threshold (until varying threshold) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 450 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 500 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 550 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 600 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 800 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 1000 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 200 gap (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 400 gap (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 600 gap (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation, 800 gap (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation augmented (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation augmented (until varying threshold) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation augmented, 400 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 50 motifs, 512 dilation augmented, 400 fusion threshold (until varying threshold) + MXfold2",
            "DivideFold CNN, 50 motifs, 1024 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 100 motifs (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 100 motifs, 256 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 128 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 100 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 200 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 300 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 400 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 500 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 600 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 700 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 800 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 900 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 1100 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 1200 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 1300 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 1400 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 1500 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 1600 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 1700 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 1800 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 1900 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 2000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until 2000 nc., 1 min steps) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation (until varying threshold) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 100 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 200 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 250 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 300 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 350 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 400 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 500 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 600 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 700 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 800 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 900 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 1100 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 1200 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 1300 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 1400 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 1500 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 1600 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 1700 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 1800 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 1900 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 2000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until 2000 nc., 1 min steps) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 fusion threshold (until varying threshold) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 450 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 500 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 550 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 600 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 800 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 1000 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 200 gap (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 400 gap (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 600 gap (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation, 800 gap (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation augmented (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation augmented (until varying threshold) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation augmented, 400 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 256 dilation augmented, 400 fusion threshold (until varying threshold) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation (until 100 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation (until 200 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation (until 400 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation (until 600 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation (until 800 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation (until varying threshold) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 100 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 200 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 250 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 300 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 350 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 400 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 400 fusion threshold (until varying threshold) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 450 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 500 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 550 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 600 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 800 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 1000 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 200 gap (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 400 gap (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 600 gap (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation, 800 gap (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation augmented (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation augmented (until varying threshold) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation augmented, 400 fusion threshold (until 1000 nc.) + MXfold2",
            "DivideFold CNN, 200 motifs, 512 dilation augmented, 400 fusion threshold (until varying threshold) + MXfold2",
            "DivideFold CNN, 293 motifs (until 1000 nc.) + MXfold2",
            "DivideFold Oracle (until 1000 nc.) + RNAfold",
            "DivideFold Oracle (until 1000 nc.) + LinearFold",
            "DivideFold Oracle (until 1000 nc.) + RNAsubopt",
            "DivideFold Oracle (until 1000 nc.) + Ensemble",
            "DivideFold Oracle (until 200 nc.) + MXfold2",
            "DivideFold Oracle (until 500 nc.) + MXfold2",
            "DivideFold Oracle (until 1000 nc.) + MXfold2",
            "DivideFold Oracle (until 1500 nc.) + MXfold2",
            "DivideFold Oracle (until 2000 nc.) + MXfold2",
            "DivideFold Oracle (until varying threshold) + MXfold2",
        ]
    ]
)

# Familywise
# data = pd.concat(
#     [
#         data_scores[data_scores.model == x]
#         for x in [
#             "RNAfold",
#             "LinearFold",
#             "MXfold2",
#             "DivideFoldCNN (until 1000 nc.) + MXfold2 (80% similarity)",
#             "DivideFoldCNN (until 1000 nc.) + MXfold2 (85% similarity)",
#             "DivideFoldCNN (until 1000 nc.) + MXfold2 (90% similarity)",
#             "DivideFoldCNN (until 1000 nc.) + MXfold2 (95% similarity)",
#             "DivideFoldCNN (until 1000 nc.) + MXfold2 (100% similarity)",
#             # "DivideFoldCNN (until 1000 nc.) + LinearFold (80% similarity)",
#             # "DivideFoldCNN (until 1000 nc.) + LinearFold (85% similarity)",
#             # "DivideFoldCNN (until 1000 nc.) + LinearFold (90% similarity)",
#             # "DivideFoldCNN (until 1000 nc.) + LinearFold (95% similarity)",
#             # "DivideFoldCNN (until 1000 nc.) + LinearFold (100% similarity)",
#             # "DivideFoldCNN (until 1000 nc.) + RNAfold (80% similarity)",
#             # "DivideFoldCNN (until 1000 nc.) + RNAfold (85% similarity)",
#             # "DivideFoldCNN (until 1000 nc.) + RNAfold (90% similarity)",
#             # "DivideFoldCNN (until 1000 nc.) + RNAfold (95% similarity)",
#             # "DivideFoldCNN (until 1000 nc.) + RNAfold (100% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + MXfold2 (80% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + MXfold2 (85% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + MXfold2 (90% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + MXfold2 (95% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + MXfold2 (100% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + LinearFold (80% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + LinearFold (85% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + LinearFold (90% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + LinearFold (95% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + LinearFold (100% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + RNAfold (80% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + RNAfold (85% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + RNAfold (90% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + RNAfold (95% similarity)",
#             # "DivideFoldMLP (until 1000 nc.) + RNAfold (100% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + MXfold2 (80% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + MXfold2 (85% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + MXfold2 (90% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + MXfold2 (95% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + MXfold2 (100% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + LinearFold (80% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + LinearFold (85% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + LinearFold (90% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + LinearFold (95% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + LinearFold (100% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + RNAfold (80% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + RNAfold (85% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + RNAfold (90% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + RNAfold (95% similarity)",
#             # "DivideFoldORACLE (until 1000 nc.) + RNAfold (100% similarity)",
#         ]
#     ]
# )


def clean_data(data):
    data = data.copy()
    if "fscore" in data.columns:
        data = data[data.fscore.notna()]
    data = data[(data.length < 1650) | ((data.length > 2750))]
    seqs_all_models = data.groupby("seq").model.nunique() == data.model.nunique()
    # data = data[(data.seq.isin(seqs_all_models[seqs_all_models].index))]
    return data


plot_all(clean_data(data))
plt.show()


def get_ranking(data, start, end, reorder=True):
    data = data.copy()
    data_view = data[(data.length >= start) & (data.length < end)]
    print(f"{start} to {end}: {data_view.rna_name.nunique()} RNAs")
    df = pd.DataFrame(
        {
            "cut_break_rate": data_view.groupby("model").cut_break_rate.mean(),
            "cut_compression": data_view.groupby("model").cut_compression.mean(),
            "fscore": data_view.groupby("model").fscore.mean(),
            "time": data_view.groupby("model").time.mean(),
        }
    )
    if df.shape[0] > 0:
        df = df.loc[data.model.unique(), :]
    if reorder:
        df = df.sort_values("fscore", ascending=False)
    df.cut_break_rate = df.cut_break_rate.apply(
        lambda x: np.nan if pd.isna(x) else str(round(100 * x, 2)) + "%"
    )
    df.cut_compression = df.cut_compression.apply(
        lambda x: np.nan if pd.isna(x) else str(round(100 * x)) + "%"
    )
    df.fscore = df.fscore.apply(lambda x: round(x, 4))
    return df


ranks = get_ranking(data, 1000, 5000)
ranks.to_excel(r"resources/results/cutting_metrics.xlsx")
# 200;400: DO NOT CUT
# 400;1300: 1 step
# 1300;1600: 200
# 1600;2000: 1 step
# 2000;4300: 2000
ranks = [
    get_ranking(data, i * 100, (i + 1) * 100, reorder=False)[["fscore"]].rename(
        columns={"fscore": f"fscore_{i*100}-{(i+1)*100}"}
    )
    for i in range(43)
]
ranks = [r for r in ranks if r.shape[0]]
ranks = pd.concat(ranks, axis=1)

X = [int(c.split("_")[1].split("-")[0]) for c in ranks.columns]
ranks = ranks.loc[[idx for idx in ranks.index if "fusion" not in idx], :]
cmap = plt.colormaps["jet"]
for i, idx in enumerate(ranks.index):
    color = cmap(i / ranks.shape[0])  # if "steps" not in idx else "black"
    plt.plot(X, ranks.loc[idx, :], color=color, label=idx)
plt.legend(loc="lower left", prop={"size": 7})
plt.show()
