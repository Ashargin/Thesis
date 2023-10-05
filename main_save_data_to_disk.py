import numpy as np
import pandas as pd
import time
import pickle
import scipy
import os
from pathlib import Path

from src.utils import format_data

# Download and prepare dataset
df_train = pd.read_csv(Path("resources/data/train.csv"), index_col=0)
df_test = pd.read_csv(Path("resources/data/test.csv"), index_col=0)
print(df_train.shape[0], "Training sequences")
print(df_test.shape[0], "Validation sequences")

files = os.listdir(Path("resources/data/formatted"))
idxs = [int(f.split(".")[0]) for f in files]
df = df_train[
    ~df_train.index.isin(idxs)
]  ################### filter over missing observations
outpath = Path("resources/data/formatted_train")  ################# adjust path
n = df.shape[0]

tstart = time.time()
bin_time = time.time()
for i, (idx, seq, cuts, outer) in enumerate(zip(df.index, df.seq, df.cuts, df.outer)):
    if i > 0:
        print(
            f"{i}/{n}, {df.index[i-1]}, average time {(time.time() - tstart) / i} total, {time.time() - bin_time} current"
        )
        bin_time = time.time()

    if len(seq) > 3400:
        continue

    seq_res, cuts_res = format_data(seq, cuts)
    outer_res = np.zeros((1,)) + outer
    res = (
        scipy.sparse.csr_matrix(seq_res),
        scipy.sparse.csr_matrix(cuts_res),
        scipy.sparse.csr_matrix(outer_res),
    )

    with open(outpath / f"{idx}.pkl", "wb") as outfile:
        pickle.dump(res, outfile, pickle.HIGHEST_PROTOCOL)

    del res
    del seq_res
