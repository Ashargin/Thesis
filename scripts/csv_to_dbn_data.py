import pandas as pd

filename = "test_sequencewise"

# Read original dbn
with open(rf"bpRNA_1m/dbnFiles/allDbn.dbn", "r") as f:
    txt = f.read()
lines = txt.strip().split("\n")
names, seqs, structs = lines[::3], lines[1::3], lines[2::3]
names = [n.split("#Name: ")[1] for n in names]

# Read target names in csv
df_in = pd.read_csv(rf"resources/data_splits/{filename}.csv", index_col=0)
target_names = list(df_in.rna_name.unique())

# Make target dbn
dbn = ""
for n, seq, struct in zip(names, seqs, structs):
    if n in target_names:
        dbn += f"#Name: {n}\n{seq}\n{struct}\n"

# Upload dbn
with open(rf"resources/data_structures/{filename}.dbn", "w") as f:
    f.write(dbn)
