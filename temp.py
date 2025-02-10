import pandas as pd

df_rfam = pd.read_csv(r"rfam_families.csv")

txt_train = open(r"resources/data_structures/train.dbn", "r").read()
lines_train = txt_train.strip().split("\n")
names, seqs, structs = lines_train[::3], lines_train[1::3], lines_train[2::3]
names = [n.split("#Name: ")[1] for n in names]
df_train = pd.DataFrame({"rna_name": names, "seq": seqs, "struct": structs})

txt_validation = open(
    r"resources/data_structures/validation_sequencewise.dbn", "r"
).read()
lines_validation = txt_validation.strip().split("\n")
names, seqs, structs = (
    lines_validation[::3],
    lines_validation[1::3],
    lines_validation[2::3],
)
names = [n.split("#Name: ")[1] for n in names]
df_validation = pd.DataFrame({"rna_name": names, "seq": seqs, "struct": structs})

txt_test = open(r"resources/data_structures/test_sequencewise.dbn", "r").read()
lines_test = txt_test.strip().split("\n")
names, seqs, structs = lines_test[::3], lines_test[1::3], lines_test[2::3]
names = [n.split("#Name: ")[1] for n in names]
df_test = pd.DataFrame({"rna_name": names, "seq": seqs, "struct": structs})

df_bprna = pd.concat([df_train, df_validation, df_test]).reset_index(drop=True)
df_norfam = df_bprna[df_bprna.rna_name.apply(lambda x: "RFAM" not in x)]

# df_rfam = df_rfam[df_rfam.rna_name.isin(df_bprna.rna_name.unique())]
df_rfam = df_rfam[df_rfam.length >= 1000]
res = df_rfam.seq.isin(df_norfam.seq.unique())

##############################
from pathlib import Path
import pandas as pd

## Read source
path_bprna = Path("bpRNA_1m")
src_dbn = open(path_bprna / "dbnFiles" / "allDbn.dbn").read()
src_lines = src_dbn.strip().split("\n")
names, seqs, structs = src_lines[::3], src_lines[1::3], src_lines[2::3]
names = [n.split("#Name: ")[1] for n in names]
base_df = pd.DataFrame({"rna_name": names, "seq": seqs, "struct": structs})

df_rfam = pd.read_csv(r"rfam_families.csv")
