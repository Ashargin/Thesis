import os
import shutil
from pathlib import Path
import pandas as pd
import re
import itertools

from src.utils import struct_to_pairs

# Read source
path_resources = Path("resources")
path_structures = path_resources / "data_structures"
path_splits = path_resources / "data_splits"
path_bprna = Path("bpRNA_1m")
src_dbn = open(path_bprna / "dbnFiles" / "allDbn.dbn").read()
src_lines = src_dbn.strip().split("\n")
names, seqs, structs = src_lines[::3], src_lines[1::3], src_lines[2::3]
names = [n.split("#Name: ")[1] for n in names]
base_df = pd.DataFrame({"rna_name": names, "seq": seqs, "struct": structs})
assert base_df.rna_name.nunique() == base_df.shape[0]

# Clean
base_df.seq = base_df.seq.apply(
    lambda x: re.sub("[^AUCG]", "N", x.upper().replace("T", "U"))
)
base_df.struct = base_df.struct.apply(
    lambda x: re.sub("[^\(\)\[\]<>\{\}AaBbCc\.]", ".", x)
)
for st in base_df.struct:
    try:
        struct_to_pairs(st)
    except Exception as e:
        raise Warning

# Read clustering, filter data
clusters_lines = open(path_resources / "clusters.clstr").read().strip().split("\n")
clusters = []
for l in clusters_lines:
    if l.startswith(">"):
        clusters.append([])
    else:
        clusters[-1].append(l.split(" ")[0])

all_names = list(itertools.chain.from_iterable(clusters))
df = base_df[base_df.rna_name.isin(all_names)]
assert df.seq.apply(len).min() > 10
assert df.seq.nunique() == df.shape[0]

# Read splits. Write structures and splits
splits_df = pd.read_csv(path_splits / "all_splits.csv", index_col=0)

# Read TR0, split train / test
path_mxfold2_data = Path("mxfold2_data")
tr0 = open(path_mxfold2_data / "TR0-canonicals.lst", "r").read().strip()
tr0_names = [l.split("/")[-1].split(".bpseq")[0] for l in tr0.split("\n")]
assert set(tr0_names).issubset(set(df.rna_name))
train_clusters = [c for c in clusters if set(c).intersection(set(tr0_names))]
test_clusters = [c for c in clusters if not set(c).intersection(set(tr0_names))]

# Balance train and test size
new_train_clusters = [c for c in train_clusters]
new_test_clusters = []
counter = 0
for i, c in enumerate(test_clusters):
    if i in [1943, 2383]:  # special huge clusters assigned to train
        new_train_clusters.append(c)
        continue
    elif i in [1957, 3577]:  # special huge clusters assigned to test
        new_test_clusters.append(c)
        continue
    elif counter % 5 == 0:
        new_test_clusters.append(c)
    else:
        new_train_clusters.append(c)
    counter += 1
train_clusters = new_train_clusters
test_clusters = new_test_clusters

train_names = list(itertools.chain.from_iterable(train_clusters))
test_names = list(itertools.chain.from_iterable(test_clusters))
train_df = df[df.rna_name.isin(train_names)]
test_df = df[df.rna_name.isin(test_names)]

# Write structures and splits
def write_files_dbn_csv_bpseq(filename, src_df, max_bpseq_length=500):
    # Write structures dbn
    with open(path_structures / f"{filename}.dbn", "w") as f:
        for _, row in src_df.iterrows():
            f.write(f"#Name: {row.rna_name}\n{row.seq}\n{row.struct}\n")

    # Write splits csv
    src_splits_df = splits_df[splits_df.rna_name.isin(src_df.rna_name.unique())]
    src_splits_df.to_csv(path_splits / f"{filename}.csv")

    # Write structures bpseq for mxfold2 training data
    # bpseq_path = path_bprna / "bpseqFiles"
    # target_dir = path_structures / filename
    # os.mkdir(target_dir)
    # os.mkdir(target_dir / "bpseqs")
    # list_file = open(target_dir / f"{filename}.lst", "w")
    # for name in src_df.rna_name.unique():
    #     src_filename = bpseq_path / (name + ".bpseq")
    #     target_filename = target_dir / "bpseqs" / (name + ".bpseq")
    #     file_lines = open(src_filename, "r").read().strip().split("\n")
    #     length = int(file_lines[-1].split(" ")[0])
    #     if length <= max_bpseq_length:
    #         shutil.copyfile(src_filename, target_filename)
    #         list_file.write(target_filename.as_posix() + "\n")
    # list_file.close()


write_files_dbn_csv_bpseq("train_sequencewise", train_df)
write_files_dbn_csv_bpseq("test_sequencewise", test_df)
