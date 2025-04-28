import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(
    r"C:\Work\Thesis\resources\data_splits\dividefold\test_sequencewise.csv"
)

a = np.cumsum(df.groupby("rna_name").size().value_counts().sort_index())
# print(a / df.rna_name.nunique())

a = df.groupby("rna_name").size().value_counts().sort_index().values
a = np.append(10174 - df.rna_name.nunique(), a)
# print(np.cumsum(a) / 10174)

df["length"] = df.seq.apply(len)
a = df.groupby("rna_name").length.sum() / df.groupby("rna_name").length.max()
a = a.sort_values()
plt.plot(
    np.arange(20), np.array([(a <= i).sum() / df.rna_name.nunique() for i in range(20)])
)
plt.plot([0, 20], [1, 1], color="black")
plt.show()

plt.plot(
    df.groupby("rna_name").length.max(),
    df.groupby("rna_name").length.sum() / df.groupby("rna_name").length.max(),
    "o",
)
plt.show()
