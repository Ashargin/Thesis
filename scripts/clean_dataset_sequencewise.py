import os
import shutil
import pandas as pd
import re

train = pd.read_csv(r"C:\Work\Thesis\resources\data_splits\train_base.csv", index_col=0)
test = pd.read_csv(r"C:\Work\Thesis\resources\data_splits\test_base.csv", index_col=0)
train.seq = train.seq.apply(
    lambda x: re.sub("[^AUCG]", "N", x.upper().replace("T", "U"))
)
test.seq = test.seq.apply(lambda x: re.sub("[^AUCG]", "N", x.upper().replace("T", "U")))
train["length"] = train.seq.apply(len)
test["length"] = test.seq.apply(len)

idx_max = train.groupby("rna_name").length.transform("max") == train.length
train = train[idx_max]

idx_max = test.groupby("rna_name").length.transform("max") == test.length
test = test[idx_max]

train = train.groupby("seq").first().reset_index()
test = test.groupby("seq").first().reset_index()
test = test[~test.seq.isin(train.seq)]

df_train = pd.read_csv(
    r"C:\Work\Thesis\resources\data_splits\train_base.csv", index_col=0
)
df_test = pd.read_csv(
    r"C:\Work\Thesis\resources\data_splits\test_base.csv", index_col=0
)
df_train = df_train[df_train.rna_name.isin(train.rna_name.unique())]
df_test = df_test[df_test.rna_name.isin(test.rna_name.unique())]

df_train.to_csv(r"C:\Work\Thesis\resources\data_splits\train_sequencewise.csv")
df_test.to_csv(r"C:\Work\Thesis\resources\data_splits\test_sequencewise.csv")

# Upload .pkl motif insertions
if not os.path.exists(r"resources/data_splits/train_sequencewise"):
    os.mkdir(rf"resources/data_splits/train_sequencewise")
if not os.path.exists(r"resources/data_splits/test_sequencewise"):
    os.mkdir(rf"resources/data_splits/test_sequencewise")
for i in df_train.index:
    try:
        shutil.copyfile(
            rf"resources/data_splits/train_base/{i}.pkl",
            rf"resources/data_splits/train_sequencewise/{i}.pkl",
        )
    except FileNotFoundError:
        continue
for i in df_test.index:
    try:
        shutil.copyfile(
            rf"resources/data_splits/test_base/{i}.pkl",
            rf"resources/data_splits/test_sequencewise/{i}.pkl",
        )
    except FileNotFoundError:
        continue
