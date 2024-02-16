import os
from pathlib import Path
import numpy as np
import pandas as pd
import re
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import struct_to_pairs

path_preds = Path("C:/Work/FoldingBenchmarkMAS/predictions/lncRNAs")
path_data = Path("C:/Work/FoldingBenchmarkMAS/data")

# Read predictions
dfs = []
for f in os.listdir(path_preds):
    txt = open(path_preds / f, "r").read()
    assert txt[-1] == "\n"
    lines = txt[:-1].split("\n")
    assert len(lines) % 3 == 0
    names, seqs, preds = lines[0::3], lines[1::3], lines[2::3]
    preds = [re.sub("[^\(\)\.]", ".", p.split(" ")[0]) for p in preds]
    preds = [p[: len(s)] for s, p in zip(seqs, preds)]
    assert all([len(s) == len(p) for s, p in zip(seqs, preds)])
    idx_valid = [i for i, p in enumerate(preds) if p.count("(") == p.count(")")]
    names = [n for i, n in enumerate(names) if i in idx_valid]
    preds = [n for i, n in enumerate(preds) if i in idx_valid]
    names = [n[1:].strip().split(" ")[0] for n in names]
    model, _ = os.path.splitext(f)
    df = pd.DataFrame({"rna_name": names, "model": model, "pred": preds})
    dfs.append(df)

df = pd.concat(dfs)
df = pd.pivot(df, columns="model", index="rna_name", values="pred")

# Read labels
path_file = (
    path_data / "lncRNAs.fasta"
    if path_preds.name == "lncRNAs"
    else path_data / "sce_genes_folded.tab"
)
txt = open(path_file, "r").read()
assert txt[-1] == "\n"
lines = (
    txt[:-1].split("\n")
    if path_preds.name == "lncRNAs"
    else list(
        itertools.chain.from_iterable([grp.split("\t") for grp in txt[:-1].split("\n")])
    )
)
assert len(lines) % 3 == 0
names, preds = lines[0::3], lines[2::3]
names = [n[1:] for n in names] if path_preds.name == "lncRNAs" else names
df.loc[:, "struct"] = ""
for n, p in zip(names, preds):
    if n == "ROX2":
        p = p[:-1]
    if n in df.index:
        df.loc[n, "struct"] = p
df["length"] = df.struct.apply(len)
banned_rnas = ["XIST"]
df = df.loc[[i for i in df.index if i not in banned_rnas], :]
df = df.loc[df.length > 1000, :]


def get_scores(df_preds):
    n = df_preds.shape[0]
    df_res = df_preds.copy().drop(["struct", "length"], axis=1)

    # Compute scores
    ppv = []
    sen = []
    fscore = []
    for method in df_preds.columns[:-2]:
        for rna_name in df_preds.index:
            y = df_preds.loc[rna_name, "struct"]
            y_hat = df_preds.loc[rna_name, method]
            if pd.isna(y_hat):
                continue

            # Remove pseudoknots
            y = re.sub("[^\(\)\.]", ".", y)
            y_hat = re.sub("[^\(\)\.]", ".", y_hat)

            assert len(y) == len(y_hat)
            y_pairs = struct_to_pairs(y)
            y_hat_pairs = struct_to_pairs(y_hat)

            tp = np.sum((y_pairs == y_hat_pairs) & (y_hat_pairs != 0))
            fp = np.sum((y_pairs != y_hat_pairs) & (y_hat_pairs != 0))
            fn = np.sum((y_pairs != y_hat_pairs) & (y_hat_pairs == 0))

            this_ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            this_sen = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            this_fscore = (
                2 * this_sen * this_ppv / (this_sen + this_ppv)
                if (this_ppv + this_sen) > 0
                else np.nan
            )
            df_res.loc[rna_name, method] = this_fscore

    return df_res


df_res = get_scores(df).fillna(0.0)
models_ranked = df_res.mean().sort_values(ascending=False)
df_res = df_res.loc[:, models_ranked.index]

plt.figure(figsize=(20, 10))
ax = sns.boxplot(data=pd.melt(df_res), x="model", y="value")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="center")
for i, model in enumerate(df_res.columns):
    plt.text(i - 0.25, 1.0, str(round(df_res.loc[:, model].mean(), 2)))
plt.ylim([0, 1.05])
plt.ylabel("F-score")
plt.title("F-score on the curated lncRNAs dataset")
plt.savefig("Test", bbox_inches="tight")
plt.show()


def round_lengths(df, n1=200, n2=400):
    df = df.copy()
    df.length = df.length.apply(
        lambda x: round(x / n1) * n1 if x < 1000 else round(x / n2) * n2
    )
    return df


df_melt = pd.melt(df_res)
df_melt["length"] = df.length.tolist() * df_melt.model.nunique()
plt.figure()
cmap = plt.get_cmap("gist_ncar")
colors = [cmap(i) for i in np.linspace(0, 1, df_melt.model.nunique())]
sns.lineplot(
    data=round_lengths(df_melt),
    x="length",
    y="value",
    hue="model",
    estimator="mean",
    palette=colors,
)
plt.xlabel("Sequence length")
plt.ylabel("F-score")
plt.title("F-score vs sequence length")
plt.ylim([0.0, 1.0])
plt.legend(loc="upper left")
plt.show()
