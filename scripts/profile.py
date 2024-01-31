import cProfile
import pandas as pd
from pathlib import Path

from src.predict import divide_predict

df_res = pd.read_csv(
    Path("resources/results/sequencewise/divide_cnn_1000_mx_sequencewise.csv")
)
df_res = df_res[["seq", "ttot"]].sort_values("ttot", ascending=False)

seq = df_res.iloc[30].seq

cProfile.run("divide_predict(seq)")
