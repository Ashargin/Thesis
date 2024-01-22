import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_scores_df

## Score predictions
results_path = Path("resources/results/sequencewise")
files = os.listdir(results_path)


def filename_to_model_name(filename):
    parts = filename.split("_")[:-1]
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
        model = f"{model_transform[parts[0]]}{parts[1].upper()} {stop} + {model_transform[parts[3]]}"
    return model


scores = [
    get_scores_df(results_path / f, name=filename_to_model_name(f)) for f in files
]

data_sequencewise = pd.concat(scores).reset_index(drop=True)


def clean_data(data):
    data = data.copy()
    data = data[data.fscore.notna()]
    data = data[
        (data.length < 1650) | ((data.length > 2750))
    ]  # & (data.length < 3800))]
    seqs_all_models = data.groupby("seq").model.nunique() == data.model.nunique()
    # data = data[(data.seq.isin(seqs_all_models[seqs_all_models].index))]
    return data


def plot(data, yvar):
    data = data.copy()
    bins = np.array(
        [200, 400, 600, 800, 1000, 1200, 1400, 1700, 2700, 3000, 3400, 3800, 4300]
    )
    data["bin"] = data.length.apply(
        lambda x: len(bins) - np.argmax(x >= bins[::-1]) - 1
    )
    plt.figure()
    ax = sns.boxplot(data, x="bin", y=yvar, hue="model")
    xtickslabels = [
        f"{str(a)}-{str(b-1)} nc.\n{data[data.bin == i].rna_name.nunique()} RNAs"
        for i, (a, b) in enumerate(zip(bins[:-1], bins[1:]))
        if i in data.bin.unique()
    ]
    ax.set_xticklabels(xtickslabels)
    ax.set_xlabel("Length")
    ax.set_ylabel(yvar.capitalize())
    ax.set_title(f"{yvar.capitalize()} vs sequence length")
    ax.legend(loc="upper left")


def plot_all(data):
    plot(data, "fscore")
    plt.ylim([0, 1])
    # plot(data[data.length >= 1000], "mcc")
    plot(data[data.length <= 1700], "time")
    plt.ylim([0, 30])
    plot(data[data.length >= 1000], "memory")
    plt.show()


data = pd.concat(
    [
        data_sequencewise[data_sequencewise.model == x]
        for x in [  # 'DivideFoldCNN (limited to 1 steps) + MXfold2',
            # 'DivideFoldCNN (limited to 2 steps) + MXfold2',
            # 'DivideFoldCNN (limited to 3 steps) + MXfold2',
            # 'DivideFoldCNN (limited to 4 steps) + MXfold2',
            # 'DivideFoldCNN (limited to 5 steps) + MXfold2',
            "DivideFoldCNN (until 200 nc.) + MXfold2",
            "DivideFoldCNN (until 400 nc.) + MXfold2",
            "DivideFoldCNN (until 600 nc.) + MXfold2",
            "DivideFoldCNN (until 800 nc.) + MXfold2",
            "DivideFoldCNN (until 1000 nc.) + MXfold2",
            # 'DivideFoldCNN (until 1000 nc.) + LinearFold',
            # 'DivideFoldCNN (until 1000 nc.) + RNAfold',
            # 'DivideFoldCNN (until 1000 nc.) + RNAsubopt',
            # 'DivideFoldCNN (until 1000 nc.) + Ensemble',
            # 'DivideFoldMLP (until 1000 nc.) + MXfold2',
            # 'DivideFoldMLP (until 1000 nc.) + LinearFold',
            # 'DivideFoldMLP (until 1000 nc.) + RNAfold',
            # 'DivideFoldMLP (until 1000 nc.) + RNAsubopt',
            # 'DivideFoldMLP (until 1000 nc.) + Ensemble',
            # 'DivideFoldORACLE (until 1000 nc.) + MXfold2',
            # 'DivideFoldORACLE (until 1000 nc.) + LinearFold',
            # 'DivideFoldORACLE (until 1000 nc.) + RNAfold',
            # 'DivideFoldORACLE (until 1000 nc.) + RNAsubopt',
            # 'DivideFoldORACLE (until 1000 nc.) + Ensemble',
            # 'MXfold2',
            # 'LinearFold',
            # 'RNAfold',
            # 'ProbKnot',
        ]
    ]
)
plot_all(clean_data(data))
plt.show()
