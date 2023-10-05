import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
txt = open(r"E:\Scripts\Thesis\rnapar_raw_data\allDbn.dbn", "r").read()
lines = txt.split("\n")[:-1]
names, seqs, structs = lines[0::3], lines[1::3], lines[2::3]

# Parse data
n_seqs = len(names)
data = []
for j, (name, struct) in enumerate(zip(names, structs)):
    print(f"Sequence {j+1}/{n_seqs}")
    depth = 0
    start = None
    for i, c in enumerate(struct):
        if c == "(":
            if depth == 0:
                start = i
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i
                data.append((name, len(struct), i - start + 1))

data_names, data_lengths, data_part_lengths = zip(*data)
df = pd.DataFrame(
    {
        "name": data_names,
        "seq_len": data_lengths,
        "part_len": data_part_lengths,
    }
)
df.part_len /= df.seq_len  # relative length

# Plot
len_cuts = [0, 500, 1000, 2000, np.inf]
for seq_min, seq_max in zip(len_cuts[:-1], len_cuts[1:]):
    sub_df = df[(df.seq_len >= seq_min) & (df.seq_len < seq_max)]
    sns.kdeplot(data=sub_df, x="part_len", label=f"{seq_min} <= len < {seq_max}")
plt.legend()
plt.show()
