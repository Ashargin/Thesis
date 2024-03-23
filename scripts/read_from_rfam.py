import pandas as pd

version = "14.10"
txt = open(rf"C:\Work\Thesis\Rfam_data\Rfam_{version}.seed", "r").read().strip()

names = []
seqs = []
structs = []
single_chars = [":", "-", ",", "~", "_"]
for l in txt.split("\n"):
    if "#=GF AC " in l:
        name = l.split(" ")[-1]
        names.append(name)
    elif "#=GC SS_cons " in l:
        struct = l.split(" ")[-1]
        for c in single_chars:
            struct = struct.replace(c, ".")
        structs.append(struct)
    elif "#=GC RF " in l:
        seq = l.split(" ")[-1].upper()
        seqs.append(seq)

# 12.2 and 14.2 : missing seqs for SEQS ['RF00210', 'RF01879']
banned_families = ["RF00210", "RF01879"]
banned_idx = [i for i, n in enumerate(names) if n in banned_families]
names = [n for i, n in enumerate(names) if i not in banned_idx]
structs = [s for i, s in enumerate(structs) if i not in banned_idx]
if version == "14.10":
    seqs = [s for i, s in enumerate(seqs) if i not in banned_idx]

df = pd.DataFrame({"family_name": names, "seq": seqs, "struct": structs})
df.to_csv(rf"C:\Work\Thesis\Rfam_data\Rfam_{version}.csv", index=False)
