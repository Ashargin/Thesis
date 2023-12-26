import pandas as pd

filename = "ufold_preds"

df = pd.read_csv(rf"C:\Work\Thesis\resources\results\{filename}.csv")

with open(r"C:\Work\Thesis\resources\data_structures\test_sequencewise.dbn", "r") as f:
    txt = f.read()
lines = txt.strip().split("\n")
names = [x.split("#Name: ")[1] for x in lines[::3]]

df = df[df.rna_name.isin(names)].reset_index(drop=True)
df.to_csv(rf"C:\Work\Thesis\resources\results\{filename}.csv", index=False)
