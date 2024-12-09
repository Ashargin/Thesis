from pathlib import Path
import pandas as pd
import os

from src.utils import get_scores_df

folder = "sequencewise"
results_path = Path("resources/results/new_results/predictions") / folder
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
        "knotfold": "KnotFold",
        "mx": "MXfold2",
        "lf": "LinearFold",
        "rnaf": "RNAfold",
        "kf": "KnotFold",
        "ipknot": "IPknot",
        "pkiss": "pKiss",
        "probknot": "ProbKnot",
        "ufold": "UFold",
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
assert data_scores.groupby("rna_name").struct.nunique().max() == 1

# Compute mean performance
metrics = ["fscore", "pk_motif_fscore", "pk_motif_sen"]
len_ranges = ["1000-4500", "1000-1600", "1600-4500"]
models = data_scores.model.unique()
ref_model = "DivideFold CNN (1000) + KnotFold"
df_ref = pd.DataFrame(
    index=data_scores.model.unique(),
    columns=[f"{metric}_{len_range}" for len_range in len_ranges for metric in metrics],
)
for model in models:
    for metric in metrics:
        for len_range in len_ranges:
            df_ref.loc[model, f"{metric}_{len_range}"] = data_scores[
                (data_scores.length >= int(len_range.split("-")[0]))
                & (data_scores.length < int(len_range.split("-")[1]))
                & (data_scores.model == model)
            ][metric].mean()
df_ref.map(lambda x: round(x, 4)).to_excel("results_data_augmentation_reference.xlsx")

# Absolute
df_abs = pd.DataFrame(
    index=data_scores.model.unique(),
    columns=[f"{metric}_{len_range}" for len_range in len_ranges for metric in metrics],
)
for model in models:
    for metric in metrics:
        for len_range in len_ranges:
            df_abs.loc[model, f"{metric}_{len_range}"] = (
                df_ref.loc[model, f"{metric}_{len_range}"]
                - df_ref.loc[ref_model, f"{metric}_{len_range}"]
            )
df_abs.map(
    lambda x: ("+" if x >= 0 else "-")
    + str(abs(round(x, 4)))
    + "0" * (6 - len(str(abs(round(x, 4)))))
).to_excel("results_data_augmentation_absolute.xlsx")

# Relative
df_rel = pd.DataFrame(
    index=data_scores.model.unique(),
    columns=[f"{metric}_{len_range}" for len_range in len_ranges for metric in metrics],
)
for model in models:
    for metric in metrics:
        for len_range in len_ranges:
            df_rel.loc[model, f"{metric}_{len_range}"] = (
                df_ref.loc[model, f"{metric}_{len_range}"]
                / df_ref.loc[ref_model, f"{metric}_{len_range}"]
                - 1
            )
df_rel.map(
    lambda x: ("+" if x >= 0 else "-")
    + str(abs(round(x * 100, 2)))
    + "0" * (3 + len(str(abs(int(x * 100)))) - len(str(abs(round(x * 100, 2)))))
    + "%"
).to_excel("results_data_augmentation_relative.xlsx")
