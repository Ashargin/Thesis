from pathlib import Path
import os
import pandas as pd

path = Path(r"C:\Work\Thesis\resources\results\predictions")

folder = "16S23S"

files = os.listdir(path / folder)
dfs = [pd.read_csv(path / folder / f) for f in files]
lens = [x.shape[0] for x in dfs]
size = max(lens)

for f, l in zip(files, lens):
    if l < size:
        print(f"{f}\t processed {l} / {size}")

if path.name == "predictions":
    print()
    unknowns = [x.pred.apply(lambda x: "?" in x).sum() for x in dfs]
    for f, u, l in zip(files, unknowns, lens):
        if u > 0:
            print(f"{f}\t missed {u} / {l}")
