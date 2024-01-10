import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils import get_scores_df

# Data source :
# https://rnacentral.org/rna/URS00000ABFE9/562
# https://rnacentral.org/rna/URS0000868550/562

## Score predictions
results_path = Path("resources/results/16S23S")
divide_cnn_mx_scores = get_scores_df(
    results_path / "divide_cnn_1000_mx_16S23S.csv", name="DivideFold CNN1D+MX"
)
divide_cnn_lf_scores = get_scores_df(
    results_path / "divide_cnn_1000_lf_16S23S.csv", name="DivideFold CNN1D+LF"
)
divide_cnn_rnaf_scores = get_scores_df(
    results_path / "divide_cnn_1000_rnaf_16S23S.csv", name="DivideFold CNN1D+RNAF"
)
divide_cnn_sub_scores = get_scores_df(
    results_path / "divide_cnn_1000_sub_16S23S.csv", name="DivideFold CNN1D+SUB"
)
divide_mlp_mx_scores = get_scores_df(
    results_path / "divide_mlp_1000_mx_16S23S.csv", name="DivideFold MLP+MX"
)
divide_mlp_lf_scores = get_scores_df(
    results_path / "divide_mlp_1000_lf_16S23S.csv", name="DivideFold MLP+LF"
)
divide_mlp_rnaf_scores = get_scores_df(
    results_path / "divide_mlp_1000_rnaf_16S23S.csv", name="DivideFold MLP+RNAF"
)
divide_mlp_sub_scores = get_scores_df(
    results_path / "divide_mlp_1000_sub_16S23S.csv", name="DivideFold MLP+SUB"
)
divide_oracle_mx_scores = get_scores_df(
    results_path / "divide_oracle_1000_mx_16S23S.csv", name="DivideFold Oracle+MX"
)
divide_oracle_lf_scores = get_scores_df(
    results_path / "divide_oracle_1000_lf_16S23S.csv", name="DivideFold Oracle+LF"
)
divide_oracle_rnaf_scores = get_scores_df(
    results_path / "divide_oracle_1000_rnaf_16S23S.csv", name="DivideFold Oracle+RNAF"
)
divide_oracle_sub_scores = get_scores_df(
    results_path / "divide_oracle_1000_sub_16S23S.csv", name="DivideFold Oracle+SUB"
)
mxfold2_scores = get_scores_df(results_path / "mxfold2_16S23S.csv", name="MXfold2")
linearfold_scores = get_scores_df(
    results_path / "linearfold_16S23S.csv", name="LinearFold"
)
rnafold_scores = get_scores_df(results_path / "rnafold_16S23S.csv", name="RNAfold")
probknot_scores = get_scores_df(results_path / "probknot_16S23S.csv", name="ProbKnot")

data = (
    pd.concat(
        [
            divide_cnn_mx_scores,
            divide_cnn_lf_scores,
            divide_cnn_rnaf_scores,
            divide_cnn_sub_scores,
            divide_mlp_mx_scores,
            divide_mlp_lf_scores,
            divide_mlp_rnaf_scores,
            divide_mlp_sub_scores,
            divide_oracle_mx_scores,
            divide_oracle_lf_scores,
            divide_oracle_rnaf_scores,
            divide_oracle_sub_scores,
            mxfold2_scores,
            linearfold_scores,
            rnafold_scores,
            probknot_scores,
        ]
    )
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
