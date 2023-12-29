import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils import get_scores_df

## Score predictions
results_path = Path("resources/results")
divide_cnn_80_scores = get_scores_df(
    results_path / "divide_cnn_1000_mx_familywise_80.csv"
)
divide_cnn_85_scores = get_scores_df(
    results_path / "divide_cnn_1000_mx_familywise_85.csv"
)
divide_cnn_90_scores = get_scores_df(
    results_path / "divide_cnn_1000_mx_familywise_90.csv"
)
divide_cnn_95_scores = get_scores_df(
    results_path / "divide_cnn_1000_mx_familywise_95.csv"
)
divide_cnn_scores = get_scores_df(results_path / "divide_cnn_1000_mx_sequencewise.csv")
divide_mlp_80_scores = get_scores_df(
    results_path / "divide_mlp_1000_mx_familywise_80.csv"
)
divide_mlp_85_scores = get_scores_df(
    results_path / "divide_mlp_1000_mx_familywise_85.csv"
)
divide_mlp_90_scores = get_scores_df(
    results_path / "divide_mlp_1000_mx_familywise_90.csv"
)
divide_mlp_95_scores = get_scores_df(
    results_path / "divide_mlp_1000_mx_familywise_95.csv"
)
divide_mlp_scores = get_scores_df(results_path / "divide_mlp_1000_mx_sequencewise.csv")
mxfold2_scores = get_scores_df(results_path / "mxfold2_sequencewise.csv")
linearfold_scores = get_scores_df(results_path / "linearfold_sequencewise.csv")
rnafold_scores = get_scores_df(results_path / "rnafold_sequencewise.csv")
probknot_scores = get_scores_df(results_path / "probknot_sequencewise.csv")
divide_cheat_scores = get_scores_df(
    results_path / "divide_cheat_1000_mx_sequencewise.csv"
)

divide_cnn_80_scores["model"] = "DivideFold CNN1D (80% max similarity)"
divide_cnn_85_scores["model"] = "DivideFold CNN1D (85% max similarity)"
divide_cnn_90_scores["model"] = "DivideFold CNN1D (90% max similarity)"
divide_cnn_95_scores["model"] = "DivideFold CNN1D (95% max similarity)"
divide_cnn_scores["model"] = "DivideFold CNN1D (sequencewise)"
divide_mlp_80_scores["model"] = "DivideFold MLP (80% max similarity)"
divide_mlp_85_scores["model"] = "DivideFold MLP (85% max similarity)"
divide_mlp_90_scores["model"] = "DivideFold MLP (90% max similarity)"
divide_mlp_95_scores["model"] = "DivideFold MLP (95% max similarity)"
divide_mlp_scores["model"] = "DivideFold MLP (sequencewise)"
mxfold2_scores["model"] = "MXfold2"
linearfold_scores["model"] = "LinearFold"
rnafold_scores["model"] = "RNAfold"
probknot_scores["model"] = "ProbKnot"
divide_cheat_scores["model"] = "DivideFold Oracle"

data_sequencewise = pd.concat(
    [
        divide_cnn_scores,
        divide_mlp_scores,
        mxfold2_scores,
        linearfold_scores,
        rnafold_scores,
        probknot_scores,
        divide_cheat_scores,
    ]
).reset_index(drop=True)
data_sequencewise.model = data_sequencewise.model.apply(
    lambda x: x.replace(" (sequencewise)", "")
)
data_familywise = pd.concat(
    [
        divide_cnn_80_scores,
        divide_cnn_85_scores,
        divide_cnn_90_scores,
        divide_cnn_95_scores,
        divide_cnn_scores,
        divide_mlp_80_scores,
        divide_mlp_85_scores,
        divide_mlp_90_scores,
        divide_mlp_95_scores,
        divide_mlp_scores,
    ]
).reset_index(drop=True)
data_familywise_mlp = data_familywise[data_familywise.model.str.contains("MLP")]
data_familywise_cnn = data_familywise[data_familywise.model.str.contains("CNN1D")]


def clean_data(data):
    data = data.copy()
    data = data[data.fscore.notna()]
    data = data[
        (data.length < 1650) | ((data.length > 2750))
    ]  # & (data.length < 3800))]
    rna_names_all_models = (
        data.groupby("rna_name").model.nunique() == data.model.nunique()
    )
    data = data[
        (data.rna_name.isin(rna_names_all_models[rna_names_all_models].index))
        | (data.length >= 1000)
    ]
    return data


# ## Plot scores
# def round_lengths(df, n1=200, n2=400):
#     df = df.copy()
#     df.length = df.length.apply(
#         lambda x: round(x / n1) * n1 if x < 1000 else round(x / n2) * n2
#     )
#     return df
#
#
# plt.figure()
# sns.lineplot(
#     data=round_lengths(data),
#     x="length",
#     y="fscore",
#     hue="model",
#     estimator="mean",
#     palette=[
#         "tab:green",
#         "firebrick",
#         "orangered",
#         "gold",
#         "tab:blue",
#         "tab:purple",
#         "tab:orange",
#     ],
# )
# plt.xlabel("Sequence length")
# plt.ylabel("F-score")
# plt.title("F-score vs sequence length")
# plt.ylim([0.0, 0.85])
# plt.show()
#
# plt.figure()
# sns.lineplot(
#     data=round_lengths(data),
#     x="length",
#     y="mcc",
#     hue="model",
#     estimator="mean",
#     palette=[
#         "tab:green",
#         "firebrick",
#         "orangered",
#         "gold",
#         "tab:blue",
#         "tab:purple",
#         "tab:orange",
#     ],
# )
# plt.xlabel("Sequence length")
# plt.ylabel("MCC")
# plt.title("MCC vs sequence length")
# plt.ylim([0., 0.8])
# plt.show()
#
# ## Plot time and memory constraints
# plt.figure()
# sns.lineplot(
#     data=round_lengths(data, n1=50, n2=50),
#     x="length",
#     y="time",
#     hue="model",
#     estimator="mean",
#     palette=[
#         "tab:green",
#         "firebrick",
#         "orangered",
#         "gold",
#         "tab:blue",
#         "tab:purple",
#         "tab:orange",
#     ],
# )
# plt.xlabel("Sequence length")
# plt.ylabel("Time (s)")
# plt.title("Computation time vs sequence length")
# plt.xlim([0, 1600])
# plt.ylim([0, 50])
# plt.show()
#
# plt.figure()
# sns.lineplot(
#     data=round_lengths(data, n1=50, n2=50),
#     x="length",
#     y="memory",
#     hue="model",
#     estimator="mean",
#     palette=[
#         "tab:green",
#         "firebrick",
#         "orangered",
#         "gold",
#         "tab:blue",
#         "tab:purple",
#         "tab:orange",
#     ],
# )
# plt.xlabel("Sequence length")
# plt.ylabel("Memory cost / total memory")
# plt.title("Memory cost vs sequence length")
# plt.show()


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
        f"{str(a)}-{str(b-1)} nc.\n{data[data.bin == i].shape[0]} RNAs"
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


plot_all(clean_data(data_sequencewise))
# plot_all(clean_data(data_familywise_mlp))
# plot_all(clean_data(data_familywise_cnn))

# TODO: nans or 0. for fscore and mcc ?
# TODO: definition of tp fp tn fn ?
