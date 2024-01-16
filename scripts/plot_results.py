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
    parts = [f for f in filename.split("_") if not f.isnumeric()]
    model_transform = {
        "divide": "DivideFold",
        "linearfold": "LinearFold",
        "mxfold2": "MXfold2",
        "probknot": "ProbKnot",
        "rnafold": "RNAfold",
    }
    res = model_transform[parts[0]] + " " + "+".join(parts[1:-1]).upper()
    res = res.strip()
    return res


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
    ]
    print(xtickslabels)
    ax.set_xticklabels(xtickslabels)
    print(ax.get_xticklabels())
    ax.set_xlabel("Length")
    ax.set_ylabel(yvar.capitalize())
    ax.set_title(f"{yvar.capitalize()} vs sequence length")


def plot_all(data):
    plot(data, "fscore")
    plot(data, "mcc")
    plot(data, "time")
    plot(data, "memory")
    plt.show()


data = pd.concat(
    [  # data_sequencewise[data_sequencewise.model == 'DivideFold MLP+RNAF'],
        # data_sequencewise[data_sequencewise.model == 'DivideFold CNN+RNAF'],
        # data_sequencewise[data_sequencewise.model == 'DivideFold ORACLE+RNAF'],
        # data_sequencewise[data_sequencewise.model == 'DivideFold MLP+LF'],
        # data_sequencewise[data_sequencewise.model == 'DivideFold CNN+LF'],
        # data_sequencewise[data_sequencewise.model == 'DivideFold ORACLE+LF'],
        # data_sequencewise[data_sequencewise.model == 'DivideFold MLP+MX'],
        data_sequencewise[data_sequencewise.model == "DivideFold CNN+MX"],
        # data_sequencewise[data_sequencewise.model == 'DivideFold ORACLE+MX'],
        # data_sequencewise[data_sequencewise.model == 'ProbKnot'],
        data_sequencewise[data_sequencewise.model == "RNAfold"],
        data_sequencewise[data_sequencewise.model == "LinearFold"],
        data_sequencewise[data_sequencewise.model == "MXfold2"],
    ]
)
plot(clean_data(data), "fscore")
plt.legend(loc="upper left")
plt.show()
