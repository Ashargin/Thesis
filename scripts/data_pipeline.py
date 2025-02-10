import os
import shutil
from pathlib import Path
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import re
import itertools
from sklearn.model_selection import train_test_split

from src.utils import struct_to_pairs

## Read source
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


## Clean
def clean_df(df_in):
    df_in = df_in.copy()
    df_in.seq = df_in.seq.apply(lambda x: re.sub("[^A-Z]", "N", x.upper()))
    df_in.struct = df_in.struct.apply(
        lambda x: re.sub("[^\(\)\[\]\{\}<>AaBbCcDd\.]", ".", x)
    )

    def remove_noncanonicals(row):
        se, st = row.seq, row.struct
        allowed_bindings = [("A", "U"), ("G", "C"), ("G", "U")]
        pairs = struct_to_pairs(st)
        for i, x in enumerate(pairs):
            j = x - 1
            if j > i and (
                (se[i], se[j]) not in allowed_bindings
                and (se[j], se[i]) not in allowed_bindings
            ):
                st = (
                    st[: min(i, j)]
                    + "."
                    + st[min(i, j) + 1 : max(i, j)]
                    + "."
                    + st[max(i, j) + 1 :]
                )
        return st

    def clean_struct_binding_chars(st):
        binding_chars = [
            ("(", ")"),
            ("[", "]"),
            ("{", "}"),
            ("<", ">"),
            ("A", "a"),
            ("B", "b"),
            ("C", "c"),
            ("D", "d"),
        ]
        temp_chars = [
            ("1", "S"),
            ("2", "T"),
            ("3", "U"),
            ("4", "V"),
            ("5", "W"),
            ("6", "X"),
            ("7", "Y"),
            ("8", "Z"),
        ]
        for b_open, b_close in binding_chars:
            assert st.count(b_open) == st.count(b_close)
        counts = [st.count(b_open) for b_open, _ in binding_chars]
        ordered_counts = len(counts) - 1 - np.argsort(counts[::-1])[::-1]
        for i, next_idx in enumerate(ordered_counts):
            if counts[next_idx] == 0:
                break
            st = st.replace(binding_chars[next_idx][0], temp_chars[i][0])
            st = st.replace(binding_chars[next_idx][1], temp_chars[i][1])
        for i in range(len(binding_chars)):
            st = st.replace(temp_chars[i][0], binding_chars[i][0])
            st = st.replace(temp_chars[i][1], binding_chars[i][1])
        return st

    df_in.struct = df_in.apply(remove_noncanonicals, axis=1)
    df_in.struct = df_in.struct.apply(clean_struct_binding_chars)
    df_in = df_in[df_in.seq.apply(len) > 10]
    return df_in


base_df = clean_df(base_df)

## Read clustering, filter data
clusters_lines = open(path_resources / "clusters.clstr").read().strip().split("\n")
clusters = []
for l in clusters_lines:
    if l.startswith(">"):
        clusters.append([])
    else:
        clusters[-1].append(l.split(" ")[0])

all_names = list(itertools.chain.from_iterable(clusters))
df = base_df[base_df.rna_name.isin(all_names)]
assert df.seq.nunique() == df.shape[0]

## Read TR0
path_mxfold2_data = Path("mxfold2_data")
tr0 = open(path_mxfold2_data / "TR0-canonicals.lst", "r").read().strip()
tr0_names = [l.split("/")[-1].split(".bpseq")[0] for l in tr0.split("\n")]
assert set(tr0_names).issubset(set(df.rna_name))

## Split train / val / test
inv_cluster = {}
for i, clus in enumerate(clusters):
    for n in clus:
        inv_cluster[n] = i
df["cluster_id"] = df.rna_name.apply(lambda x: inv_cluster[x])
df["is_in_tr0"] = df.rna_name.isin(tr0_names)

tr0_clusters = df.loc[df["is_in_tr0"], "cluster_id"].unique()
not_tr0_clusters = set(df["cluster_id"].unique()) - set(tr0_clusters)
tr0_clusters = sorted(tr0_clusters)
not_tr0_clusters = sorted(not_tr0_clusters)
n_tr0_seqs = df["cluster_id"].isin(tr0_clusters).sum()
n_not_tr0_seqs = df["cluster_id"].isin(not_tr0_clusters).sum()

target_train_size = 0.8
train_size = int(target_train_size * len(df)) - n_tr0_seqs
train_size /= n_not_tr0_seqs

train_clusters, test_clusters = train_test_split(
    not_tr0_clusters, train_size=train_size, shuffle=True, random_state=0
)
train_clusters = tr0_clusters + train_clusters

target_val_size = 0.1  # relatively to test

train_df = df[df.cluster_id.isin(train_clusters)].drop(
    ["cluster_id", "is_in_tr0"], axis=1
)
test_df = df[df.cluster_id.isin(test_clusters)].drop(
    ["cluster_id", "is_in_tr0"], axis=1
)
validation_df = test_df.sample(frac=0.2, random_state=0).sort_index()
test_df = test_df[~test_df.rna_name.isin(validation_df.rna_name.unique())]

## Get family-wise dataset from RFAM
def get_rfam_df(version):
    txt = (
        open(Path(f"Rfam_data/Rfam_{version}.seed"), "r", encoding="latin-1")
        .read()
        .strip()
    )
    family_names = []
    names = []
    seqs = []
    structs = []
    single_chars = [":", "-", ",", "~", "_"]
    # forbidden_bindings_chars = ["A", "a", "B", "b", "C", "c", "D", "d"]
    for block in txt.split("#=GF AC ")[1:]:
        fam_name = block.strip().split("\n")[0]
        block_names = []
        block_seqs = []
        ref_struct = ""
        for l in block.split("\n"):
            if (
                l
                and not l.startswith("#")
                and not l.startswith(" ")
                and not l.startswith("/")
            ):
                block_names.append(l.split(" ")[0])
                block_seqs.append(l.split(" ")[-1])
            elif l.startswith("#=GC SS_cons "):
                ref_struct = l.split(" ")[-1]
                for c in single_chars:
                    ref_struct = ref_struct.replace(c, ".")

        assert all([len(seq) == len(ref_struct) for seq in block_seqs])

        family_names += [fam_name] * len(block_seqs)
        names += block_names
        ref_pairs = struct_to_pairs(ref_struct)
        aligned_seqs = []
        aligned_structs = []
        for seq in block_seqs:
            this_struct = ref_struct
            # for c in forbidden_bindings_chars:
            #     this_struct = this_struct.replace(c, ".")
            for i, c in enumerate(seq):
                if ref_pairs[i] > 0 and c == "-":
                    j = ref_pairs[i] - 1
                    this_struct = (
                        this_struct[: min(i, j)]
                        + "."
                        + this_struct[min(i, j) + 1 : max(i, j)]
                        + "."
                        + this_struct[max(i, j) + 1 :]
                    )

            this_struct = "".join(
                [c_st for c_se, c_st in zip(seq, this_struct) if c_se != "-"]
            )
            aligned_seqs.append(seq.replace("-", ""))
            aligned_structs.append(this_struct)

        seqs += aligned_seqs
        structs += aligned_structs

    rfam_df = pd.DataFrame(
        {"family_name": family_names, "rna_name": names, "seq": seqs, "struct": structs}
    )
    rfam_df.drop_duplicates("rna_name", inplace=True)
    rfam_df = clean_df(rfam_df)
    return rfam_df


rfam_122_df = get_rfam_df("12.2")
# rfam_142_df = get_rfam_df("14.2")
# rfam_1410_df = get_rfam_df("14.10")
rfam_150_df = get_rfam_df("15.0")
rfam_test_df = (
    rfam_150_df[~rfam_150_df.family_name.isin(rfam_122_df.family_name.unique())]
    .drop("family_name", axis=1)
    .reset_index(drop=True)
)

## Read RFAM clusters and deduplicate
clusters_lines = (
    open(path_resources / "clusters_rfam_15.0.clstr").read().strip().split("\n")
)
rfam_clusters = []
for l in clusters_lines:
    if l.startswith(">"):
        rfam_clusters.append([])
    else:
        rfam_clusters[-1].append(l.split(" ")[0])

head_names = [clus[0] for clus in rfam_clusters]
assert set(head_names).issubset(set(rfam_test_df.rna_name))
rfam_test_df = rfam_test_df[rfam_test_df.rna_name.isin(head_names)]
assert rfam_test_df.seq.nunique() == rfam_test_df.shape[0]

## Read splits. Write structures and splits
splits_df = pd.read_csv(path_splits / "all_splits.csv")


## CD-HIT clusters for RFAM and finish family-wise
def write_files_dbn_csv_bpseq(
    filename, src_df, write_splits=True, max_bpseq_length=500
):
    # Write structures dbn
    with open(path_structures / f"{filename}.dbn", "w") as f:
        for _, row in src_df.iterrows():
            f.write(f"#Name: {row.rna_name}\n{row.seq}\n{row.struct}\n")

    # Write splits csv
    if write_splits:
        src_splits_df = splits_df[splits_df.rna_name.isin(src_df.rna_name.unique())]
        src_splits_df.reset_index(drop=True).to_csv(path_splits / f"{filename}.csv")

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


write_files_dbn_csv_bpseq("train", train_df)
write_files_dbn_csv_bpseq("validation", validation_df)
write_files_dbn_csv_bpseq("test_sequencewise", test_df)
write_files_dbn_csv_bpseq("test_familywise15", rfam_test_df, write_splits=False)
