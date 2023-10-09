import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

rna = "CRW_1"

# bpseqFiles
skiprows = 1 * ("RNP" in rna or "SPR" in rna or "tmRNA" in rna) + 2 * ("CRW" in rna)
df = pd.read_csv(
    Path(f"bpRNA_1m/bpseqFiles/bpRNA_{rna}.bpseq"),
    sep=" ",
    skiprows=skiprows,
    header=None,
    names=["id", "nucl", "pair"],
).set_index("id")

# staFiles
sta = pd.read_csv(Path(f"bpRNA_1m/staFiles/bpRNA_{rna}.sta"), skiprows=2, header=None).T
sta.columns = ["sequence", "pair_type", "structure", "pseudoknot"]

for c in sta.columns[1:]:
    df[c] = list(sta[c].iloc[0])


def df_to_contact_matrix(df):
    seq_len = df.shape[0]
    mat = np.zeros((seq_len, seq_len, 2), dtype=int)

    df_bp = df[df.pair_type.isin(["(", ")"])]
    df_pk = df[~df.pair_type.isin([".", "(", ")"])]
    assert np.all(df_bp.pseudoknot == "N")
    assert np.all(df_pk.pseudoknot == "K")

    mat[df_bp.index - 1, df_bp.pair - 1, 0] = 1
    mat[df_pk.index - 1, df_pk.pair - 1, 1] = 1
    assert np.all(mat == np.swapaxes(mat, 0, 1))

    return mat


def df_to_structure_matrix(df):
    seq_len = df.shape[0]
    mat = np.zeros((seq_len, seq_len, 2), dtype=int)

    df_bp = df[df.pair_type.isin(["(", ")"])]
    df_pk = df[~df.pair_type.isin([".", "(", ")"])]
    assert np.all(df_bp.pseudoknot == "N")
    assert np.all(df_pk.pseudoknot == "K")

    mat[df_bp.index - 1, df_bp.pair - 1, 0] = 1
    mat[df_pk.index - 1, df_pk.pair - 1, 1] = 1
    assert np.all(mat == np.swapaxes(mat, 0, 1))

    return mat


def plot_pairs(mat):
    plt.spy(mat[:, :, 0], color="lightskyblue", markersize=1)
    plt.spy(mat[:, :, 1], color="orangered", markersize=3)
    plt.show()


mat = df_to_contact_matrix(df)
plot_pairs(mat)

# Info in :
# - letters (ignore ?)
# - contacts
# - structure => todo
