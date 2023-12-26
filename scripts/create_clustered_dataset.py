import shutil
import os
import pandas as pd

cluster_threshold = 95

# Read clusters
with open(f"cdhit_clusters_{cluster_threshold}.fasta", "r") as f:
    txt = f.read()
lines = txt.strip().split("\n")
names = [x[1:] for x in lines[::2]]

# Read base split
df_train = pd.read_csv(r"resources/data/train_base.csv", index_col=0)
df_test = pd.read_csv(r"resources/data/test_base.csv", index_col=0)
old_train_size = round(df_train.shape[0] / (df_train.shape[0] + df_test.shape[0]), 2)
print(f"Old train size: {old_train_size}")

# Make familywise split
df_train = df_train[df_train.rna_name.isin(names)]
df_test = df_test[df_test.rna_name.isin(names)]
new_train_size = round(df_train.shape[0] / (df_train.shape[0] + df_test.shape[0]), 2)
print(f"New train size: {new_train_size}")

if new_train_size >= 0.78 and new_train_size <= 0.82:
    # Upload csv
    df_train.to_csv(rf"resources/data/train_familywise_{cluster_threshold}.csv")
    df_test.to_csv(rf"resources/data/test_familywise_{cluster_threshold}.csv")

    # Upload .pkl motif insertions
    if not os.path.exists(rf"resources/data/train_familywise_{cluster_threshold}"):
        os.mkdir(rf"resources/data/train_familywise_{cluster_threshold}")
    if not os.path.exists(rf"resources/data/test_familywise_{cluster_threshold}"):
        os.mkdir(rf"resources/data/test_familywise_{cluster_threshold}")
    for i in df_train.index:
        try:
            shutil.copyfile(
                rf"resources/data/train_base/{i}.pkl",
                rf"resources/data/train_familywise_{cluster_threshold}/{i}.pkl",
            )
        except FileNotFoundError:
            continue
    for i in df_test.index:
        try:
            shutil.copyfile(
                rf"resources/data/test_base/{i}.pkl",
                rf"resources/data/test_familywise_{cluster_threshold}/{i}.pkl",
            )
        except FileNotFoundError:
            continue
