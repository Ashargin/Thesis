import os
from pathlib import Path
import re
import numpy as np

path_bprna_new = Path("mxfold2_data/bpRNAnew_dataset/bpRNAnew.nr500.canonicals")
files = os.listdir(path_bprna_new)
files = [f for f in files if not f.startswith("._")]
seqs = []
pairs = []
for f in files:
    lines = open(path_bprna_new / f, "r").read().strip().split("\n")
    seq = "".join([l.split(" ")[1] for l in lines])
    this_pairs = [int(l.split(" ")[2]) for l in lines]
    seqs.append(seq)
    pairs.append(np.array(this_pairs))

names = [
    "_".join(f.split(".bpseq")[0].split("_")[:-1])
    + "/"
    + f.split(".bpseq")[0].split("_")[-1]
    for f in files
]
df_bprna_new = pd.DataFrame({"rna_name": names, "seq_new": seqs, "pairs_new": pairs})

df_rfam.set_index("rna_name", inplace=True)
df_bprna_new.set_index("rna_name", inplace=True)
df = df_rfam.join(df_bprna_new, how="inner")
