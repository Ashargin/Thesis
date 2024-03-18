import os
import shutil
from pathlib import Path
import pandas as pd
import re

from src.utils import struct_to_pairs

# Read source
path_structures = Path("resources/data_structures")
path_splits = Path("resources/data_splits")
src_dbn = open(Path("bpRNA_1m/dbnFiles/allDbn.dbn")).read()
src_lines = src_dbn.strip().split("\n")
names, seqs, structs = src_lines[::3], src_lines[1::3], src_lines[2::3]
base_df = pd.DataFrame({"rna_name": names, "seq": seqs, "struct": structs})

# Clean
base_df.rna_name = base_df.rna_name.apply(lambda x: x.split("#Name: ")[1])
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

# Read train/test sequence-wise and family-wise split data
def read_df_from_dbn(filename):
    txt = open(path_structures / f"{filename}.dbn", "r").read()
    lines = txt.strip().split("\n")
    names = [n.split("#Name: ")[1] for n in lines[::3]]
    this_df = base_df[base_df.rna_name.isin(names)].sort_index()
    return this_df


train_seq_df = read_df_from_dbn("train_sequencewise")
test_seq_df = read_df_from_dbn("test_sequencewise")
train_fam_df = read_df_from_dbn("train_familywise")
test_fam_df = read_df_from_dbn("test_familywise")
df = pd.concat([train_seq_df, test_seq_df]).sort_index()

# Write fasta
with open(path_structures / "all_fasta.fasta", "w") as f:
    for name, seq in zip(df.rna_name, df.seq):
        f.write(f">{name}\n{seq}\n")

# Read splits and clusters. Write structures and splits for sequence-wise and family-wise splits
splits_df = pd.read_csv(path_splits / "all_splits.csv", index_col=0)


def write_files_csv(filename, src_df):
    # Write splits csv
    src_splits_df = splits_df[splits_df.rna_name.isin(src_df.rna_name.unique())]
    src_splits_df.to_csv(path_splits / f"{filename}.csv")


write_files_csv("train_sequencewise", train_seq_df)
write_files_csv("test_sequencewise", test_seq_df)
write_files_csv("train_familywise", train_fam_df)
write_files_csv("test_familywise", test_fam_df)

# Write bpseq for mxfold2 training
for filename in [
    "train_sequencewise",
    "test_sequencewise",
    "train_familywise",
    "test_familywise",
]:
    bpseq_path = Path("bpRNA_1m/bpseqFiles")
    lines = open(path_structures / (filename + ".dbn")).read().strip().split("\n")
    names, seqs, structs = lines[::3], lines[1::3], lines[2::3]
    target_dir = path_structures / filename
    os.mkdir(target_dir)
    os.mkdir(target_dir / "bpseqs")
    list_file = open(target_dir / "filelist.lst", "w")
    for name in names:
        name = name.split("#Name: ")[1]
        src_filename = bpseq_path / (name + ".bpseq")
        target_filename = target_dir / "bpseqs" / (name + ".bpseq")

        file_lines = open(src_filename, "r").read().strip().split("\n")
        length = int(file_lines[-1].split(" ")[0])
        if length <= 500:
            shutil.copyfile(src_filename, target_filename)
            list_file.write(target_filename.as_posix() + "\n")
    list_file.close()
