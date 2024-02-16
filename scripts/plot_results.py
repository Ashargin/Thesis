import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_scores_df

## Score predictions
results_path = Path("resources/results/cutting_metrics")
files = os.listdir(results_path)


def filename_to_model_name(filename):
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
        stop = (
            f"(until {parts[2]} nc.)"
            if parts[2].isnumeric()
            else f"(limited to {parts[2][:-1]} steps)"
        )
        model_parts = re.findall(r"[^\W\d_]+|\d+", parts[1])
        model_parts = [model_parts[0].upper()] + [
            f"{n} {s}" for n, s in zip(model_parts[1::2], model_parts[2::2])
        ]
        cut_model = (
            ", ".join(model_parts)
            .replace("ORACLE", "Oracle")
            .replace("dil", "dilation")
            .replace("INV", " reversed")
        )
        model = f"{model_transform[parts[0]]} {cut_model} {stop}"
        if len(parts) > 3:
            model = model + f" + {parts[3]}"
    model = model + suffix

    return model


scores = (
    [
        get_scores_df(results_path / f).assign(model=filename_to_model_name(f))
        for f in files
    ]
    if results_path.name != "cutting_metrics"
    else [
        pd.read_csv(results_path / f).assign(model=filename_to_model_name(f))
        for f in files
    ]
)

data_scores = pd.concat(scores).reset_index(drop=True)
if "length" not in data_scores.columns:
    data_scores["length"] = data_scores.seq.apply(len)


def plot(data, yvar):
    data = data[data.length >= 1000].copy()
    bins = np.array(
        [200, 400, 600, 800, 1000, 1200, 1400, 1700, 2700, 3000, 3400, 3800, 4400]
    )
    data["bin"] = data.length.apply(
        lambda x: len(bins) - np.argmax(x >= bins[::-1]) - 1
    )
    plt.figure()
    ax = sns.boxplot(data, x="bin", y=yvar, hue="model", palette="tab20")
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
    if results_path.name != "cutting_metrics":
        plot(data, "fscore")
        plt.ylim([0, 1])
        # plot(data, "mcc")
        plot(data, "time")
        plt.ylim([0, 200])
        plot(data, "memory")
    else:
        plot(data, "break_rate")
        plt.ylim([0, 1])
        plot(data, "compression")
    plt.show()


# Sequencewise
# data = pd.concat(
#     [
#         data_scores[data_scores.model == x]
#         for x in [
#             # 'DivideFold CNN (limited to 1 steps) + MXfold2',
#             # 'DivideFold CNN (limited to 2 steps) + MXfold2',
#             # 'DivideFold CNN (limited to 3 steps) + MXfold2',
#             # 'DivideFold CNN (limited to 4 steps) + MXfold2',
#             # 'DivideFold CNN (limited to 5 steps) + MXfold2',
#             # "DivideFold CNN (until 200 nc.) + MXfold2",
#             # "DivideFold CNN (until 400 nc.) + MXfold2",
#             # "DivideFold CNN (until 600 nc.) + MXfold2",
#             # "DivideFold CNN (until 800 nc.) + MXfold2",
#             # 'DivideFold CNN, 0 motifs, (until 1000 nc.) + MXfold2',
#             # 'DivideFold CNN, 10 motifs, (until 1000 nc.) + MXfold2',
#             # 'DivideFold CNN, 25 motifs, (until 1000 nc.) + MXfold2',
#             'DivideFold CNN, 50 motifs (until 1000 nc.) + MXfold2',
#             # 'DivideFold CNN, 100 motifs, (until 1000 nc.) + MXfold2',
#             # 'DivideFold CNN, 200 motifs, (until 1000 nc.) + MXfold2',
#             # 'DivideFold CNN, 293 motifs, (until 1000 nc.) + MXfold2',
#             # 'DivideFold CNN, 50 motifs, 1 dilation (until 1000 nc.) + MXfold2',
#             # 'DivideFold CNN, 50 motifs, 2 dilation (until 1000 nc.) + MXfold2',
#             # 'DivideFold CNN, 50 motifs, 4 dilation (until 1000 nc.) + MXfold2',
#             # 'DivideFold CNN, 50 motifs, 8 dilation (until 1000 nc.) + MXfold2',
#             # 'DivideFold CNN, 50 motifs, 16 dilation (until 1000 nc.) + MXfold2',
#             'DivideFold CNN, 50 motifs, 1 dilation reversed (until 1000 nc.) + MXfold2',
#             'DivideFold CNN, 50 motifs, 2 dilation reversed (until 1000 nc.) + MXfold2',
#             'DivideFold CNN, 50 motifs, 4 dilation reversed (until 1000 nc.) + MXfold2',
#             'DivideFold CNN, 50 motifs, 8 dilation reversed (until 1000 nc.) + MXfold2',
#             'DivideFold CNN, 50 motifs, 16 dilation reversed (until 1000 nc.) + MXfold2',
#             'DivideFold CNN, 50 motifs, 32 dilation reversed (until 1000 nc.) + MXfold2',
#             'DivideFold CNN, 50 motifs, 64 dilation reversed (until 1000 nc.) + MXfold2',
#             'DivideFold CNN, 50 motifs, 128 dilation reversed (until 1000 nc.) + MXfold2',
#             'DivideFold CNN, 50 motifs, 256 dilation reversed (until 1000 nc.) + MXfold2',
#             'DivideFold CNN, 50 motifs, 512 dilation reversed (until 1000 nc.) + MXfold2',
#             'DivideFold CNN, 50 motifs, 1024 dilation reversed (until 1000 nc.) + MXfold2',
#             # 'DivideFold CNN (until 1000 nc.) + LinearFold',
#             # 'DivideFold CNN (until 1000 nc.) + RNAfold',
#             # 'DivideFold CNN (until 1000 nc.) + RNAsubopt',
#             # 'DivideFold CNN (until 1000 nc.) + Ensemble',
#             # 'DivideFold MLP (until 1000 nc.) + MXfold2',
#             # 'DivideFold MLP (until 1000 nc.) + LinearFold',
#             # 'DivideFold MLP (until 1000 nc.) + RNAfold',
#             # 'DivideFold MLP (until 1000 nc.) + RNAsubopt',
#             # 'DivideFold MLP (until 1000 nc.) + Ensemble',
#             # 'DivideFold Oracle (until 1000 nc.) + RNAfold',
#             # 'DivideFold Oracle (until 1000 nc.) + LinearFold',
#             # 'DivideFold Oracle (until 1000 nc.) + RNAsubopt',
#             # 'DivideFold Oracle (until 1000 nc.) + Ensemble',
#             # "DivideFold CNN (until 1000 nc.) + MXfold2",
#             # 'DivideFold Oracle (until 1000 nc.) + MXfold2',
#             # 'RNAfold',
#             # 'LinearFold',
#             # 'MXfold2',
#             # 'ProbKnot',
#         ]
#     ]
# )

# Familywise
# data = pd.concat(
#     [
#         data_scores[data_scores.model == x]
#         for x in [
#             'RNAfold',
#             'LinearFold',
#             'MXfold2',
#             'DivideFoldCNN (until 1000 nc.) + MXfold2 (80% similarity)',
#             'DivideFoldCNN (until 1000 nc.) + MXfold2 (85% similarity)',
#             'DivideFoldCNN (until 1000 nc.) + MXfold2 (90% similarity)',
#             'DivideFoldCNN (until 1000 nc.) + MXfold2 (95% similarity)',
#             'DivideFoldCNN (until 1000 nc.) + MXfold2 (100% similarity)',
#             # 'DivideFoldCNN (until 1000 nc.) + LinearFold (80% similarity)',
#             # 'DivideFoldCNN (until 1000 nc.) + LinearFold (85% similarity)',
#             # 'DivideFoldCNN (until 1000 nc.) + LinearFold (90% similarity)',
#             # 'DivideFoldCNN (until 1000 nc.) + LinearFold (95% similarity)',
#             # 'DivideFoldCNN (until 1000 nc.) + LinearFold (100% similarity)',
#             # 'DivideFoldCNN (until 1000 nc.) + RNAfold (80% similarity)',
#             # 'DivideFoldCNN (until 1000 nc.) + RNAfold (85% similarity)',
#             # 'DivideFoldCNN (until 1000 nc.) + RNAfold (90% similarity)',
#             # 'DivideFoldCNN (until 1000 nc.) + RNAfold (95% similarity)',
#             # 'DivideFoldCNN (until 1000 nc.) + RNAfold (100% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + MXfold2 (80% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + MXfold2 (85% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + MXfold2 (90% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + MXfold2 (95% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + MXfold2 (100% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + LinearFold (80% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + LinearFold (85% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + LinearFold (90% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + LinearFold (95% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + LinearFold (100% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + RNAfold (80% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + RNAfold (85% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + RNAfold (90% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + RNAfold (95% similarity)',
#             # 'DivideFoldMLP (until 1000 nc.) + RNAfold (100% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + MXfold2 (80% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + MXfold2 (85% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + MXfold2 (90% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + MXfold2 (95% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + MXfold2 (100% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + LinearFold (80% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + LinearFold (85% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + LinearFold (90% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + LinearFold (95% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + LinearFold (100% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + RNAfold (80% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + RNAfold (85% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + RNAfold (90% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + RNAfold (95% similarity)',
#             # 'DivideFoldORACLE (until 1000 nc.) + RNAfold (100% similarity)',
#         ]
#     ]
# )

# Cutting metrics
data = pd.concat(
    [
        data_scores[data_scores.model == x]
        for x in [
            "DivideFold CNN (until 1000 nc.)",
            "DivideFold CNN, 0 motifs (until 1000 nc.)",
            "DivideFold CNN, 10 motifs (until 1000 nc.)",
            "DivideFold CNN, 25 motifs (until 1000 nc.)",
            "DivideFold CNN, 50 motifs (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 1 dilation (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 2 dilation (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 4 dilation (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 8 dilation (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 16 dilation (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 1 dilation reversed (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 2 dilation reversed (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 4 dilation reversed (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 8 dilation reversed (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 16 dilation reversed (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 32 dilation reversed (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 64 dilation reversed (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 128 dilation reversed (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 256 dilation reversed (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 512 dilation reversed (until 1000 nc.)",
            "DivideFold CNN, 50 motifs, 1024 dilation reversed (until 1000 nc.)",
            "DivideFold CNN, 100 motifs (until 1000 nc.)",
            "DivideFold CNN, 200 motifs (until 1000 nc.)",
            "DivideFold CNN, 293 motifs (until 1000 nc.)",
            "DivideFold MLP (until 1000 nc.)",
            "DivideFold Oracle (until 1000 nc.)",
        ]
    ]
)


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
