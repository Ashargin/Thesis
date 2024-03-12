import os
import shutil
from pathlib import Path
import pandas as pd
import re

from src.utils import struct_to_pairs
from src.predict import oracle_get_cuts

# Read source
path_structures = Path("resources/data_structures")
path_splits = Path("resources/data_splits")
src_dbn = open(Path("bpRNA_1m/dbnFiles/allDbn.dbn")).read()
src_lines = src_dbn.strip().split("\n")
names, seqs, structs = src_lines[::3], src_lines[1::3], src_lines[2::3]
df = pd.DataFrame({"rna_name": names, "seq": seqs, "struct": structs})

# Clean
df.rna_name = df.rna_name.apply(lambda x: x.split("#Name: ")[1])
df.seq = df.seq.apply(lambda x: re.sub("[^AUCG]", "N", x.upper().replace("T", "U")))
df.struct = df.struct.apply(lambda x: re.sub("[^\(\)\[\]<>\{\}AaBbCc\.]", ".", x))
for st in df.struct:
    try:
        struct_to_pairs(st)
    except Exception as e:
        raise Warning

# Read train/test split data
train = open(path_structures / "train_sequencewise.dbn", "r").read()
train_lines = train.strip().split("\n")
test = open(path_structures / "test_sequencewise.dbn", "r").read()
test_lines = test.strip().split("\n")
train_names = [n.split("#Name: ")[1] for n in train_lines[::3]]
test_names = [n.split("#Name: ")[1] for n in test_lines[::3]]

# Split / deduplicate
train_df = df[df.rna_name.isin(train_names)].sort_index()
test_df = df[df.rna_name.isin(test_names)].sort_index()
df = pd.concat([train_df, test_df]).sort_index()

# Write fasta
with open(path_structures / "all_fasta.fasta", "w") as f:
    for name, seq in zip(df.rna_name, df.seq):
        f.write(f">{name}\n{seq}\n")

# Clusterize with CD-HIT
# ... use CD-HIT

# Read clusters. Write dbn for sequence-wise and family-wise splits
def write_files_dbn_csv(filename, src_df, write_dbn=True):
    if write_dbn:
        with open(path_structures / f"{filename}.dbn", "w") as f:
            for name, seq, struct in zip(src_df.rna_name, src_df.seq, src_df.struct):
                f.write(f"#Name: {name}\n{seq}\n{struct}\n")
    ser_cuts = []
    ser_outer = []
    for struct in src_df.struct:
        cuts, outer = oracle_get_cuts(struct)
        ser_cuts.append(str(cuts).replace(", ", " "))
        ser_outer.append(outer)
    src_df = src_df.copy()
    src_df["cuts"] = ser_cuts
    src_df["outer"] = ser_outer
    src_df.to_csv(path_splits / f"{filename}.csv")


clusters_path = Path("resources/data_clusters")
for filename in os.listdir(clusters_path):
    if filename.endswith(".fasta"):
        lines = open(clusters_path / filename, "r").read().strip().split("\n")
        names = [x[1:] for x in lines[::2]]
        train_cluster_df = train_df[train_df.rna_name.isin(names)]
        test_cluster_df = test_df[test_df.rna_name.isin(names)]
        similarity = int(re.search("\d+", filename).group(0))
        write_files_dbn_csv(f"train_familywise_{similarity}", train_cluster_df)
        write_files_dbn_csv(f"test_familywise_{similarity}", test_cluster_df)
write_files_dbn_csv("train_sequencewise", train_df, write_dbn=False)
write_files_dbn_csv("test_sequencewise", test_df, write_dbn=False)

# Write bpseq for mxfold2 training
for filename in os.listdir(path_structures):
    if filename in [
        "16S23S.dbn",
        "benchmark_lncRNAs.dbn",
        "benchmark_sce.dbn",
    ] or not filename.endswith(
        ".dbn"
    ):  # skipped
        continue

    bpseq_path = Path("bpRNA_1m/bpseqFiles")
    lines = open(path_structures / filename).read().strip().split("\n")
    names, seqs, structs = lines[::3], lines[1::3], lines[2::3]
    target_dir = path_structures / filename.split(".dbn")[0]
    os.mkdir(target_dir)
    os.mkdir(target_dir / "bpseqs")
    list_file = open(target_dir / "filelist.lst", "w")
    for name in names:
        name = name.split("#Name: ")[1]
        src_filename = bpseq_path / (name + ".bpseq")
        target_filename = target_dir / "bpseqs" / (name + ".bpseq")

        file_lines = open(src_filename, "r").read().strip().split("\n")
        length = int(file_lines[-1].split(" ")[0])
        if length <= 2000:
            shutil.copyfile(src_filename, target_filename)
            list_file.write(target_filename.as_posix() + "\n")
    list_file.close()
