import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_scores_df

# Data source :
# https://rnacentral.org/rna/URS00000ABFE9/562
# https://rnacentral.org/rna/URS0000868550/562

## Score predictions
results_path = Path("resources/results/16S23S")
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
    return res


scores = [
    get_scores_df(results_path / f, name=filename_to_model_name(f)) for f in files
]

data = (
    pd.concat(scores)
    .sort_values("fscore", ascending=False)
    .reset_index(drop=True)
    .loc[:, ["rna_name", "model", "fscore"]]
)
data.fscore = data.fscore.apply(lambda x: round(x, 3))
df16 = data[data.rna_name == "16S"].set_index("model").drop("rna_name", axis=1)
df16.columns = ["fscore_16S"]
df23 = data[data.rna_name == "23S"].set_index("model").drop("rna_name", axis=1)
df23.columns = ["fscore_23S"]
df = pd.concat([df16, df23], axis=1)
df.to_excel("16S23S.xlsx")
