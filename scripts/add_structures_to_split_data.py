import re
import os
from pathlib import Path
import pandas as pd
import numpy as np

path_in = Path("resources/data_splits/train_familywise_95")
print(path_in)

path_df_in = Path("resources/data_splits") / (path_in.name + ".csv")
df_in = pd.read_csv(path_df_in, index_col=0)

path_structures = Path("resources/data_structures") / (path_in.name + ".dbn")
txt_structures = open(path_structures, "r").read()
lines_structures = txt_structures.strip().split("\n")
df_structures = pd.DataFrame(
    {
        "rna_name": lines_structures[::3],
        "seq": lines_structures[1::3],
        "struct": lines_structures[2::3],
    }
)
df_structures.rna_name = df_structures.rna_name.apply(lambda x: x.split("#Name: ")[1])

structs = []
for name, seq in zip(df_in.rna_name, df_in.seq):
    other_seq = df_structures.loc[df_structures.rna_name == name, "seq"].iloc[0]
    other_struct = df_structures.loc[df_structures.rna_name == name, "struct"].iloc[0]
    if seq not in other_seq:
        structs.append(np.nan)
        continue

    regex = re.compile("(?=(" + seq + "))")
    matches = []
    for match in regex.finditer(other_seq):
        start, end = match.span(1)
        matches.append(other_struct[start:end])
    matches = [st for st in matches if st.count("(") == st.count(")")]
    if len(matches) != 1:
        structs.append(np.nan)
        continue
    assert len(matches) == 1
    struct = matches[0]
    assert struct.count("(") == struct.count(")")
    structs.append(struct)

df_in["struct"] = structs
df_in = df_in.loc[:, ["rna_name", "seq", "struct", "cuts", "outer"]]
assert np.all(
    df_in[df_in.struct.notna()].struct.apply(len)
    == df_in[df_in.struct.notna()].seq.apply(len)
)
print("STRUCTURES ADDED")
print(df_in.struct.isna().sum() / df_in.shape[0])
assert df_in.struct.isna().sum() / df_in.shape[0] < 0.03
for struct, idx in zip(df_in.struct, df_in.index):
    path_file = Path(path_in) / f"{idx}.pkl"
    if pd.isna(struct) and path_file.exists():
        os.remove(path_file)
print("FILES REMOVED")
df_in.to_csv(path_df_in)
print("SAVED")
