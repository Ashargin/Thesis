import pandas as pd
from pathlib import Path
import re


def clean_sequences(seqs):
    seqs = seqs.apply(lambda x: re.sub("[^AUCG]", "N", x))
    seqs = pd.Series(seqs.unique())
    return seqs


seqs_ribonanza = pd.read_csv(
    Path("stanford_ribonenza_data/train_data.csv"), usecols=["sequence"]
).sequence
seqs_ribonanza = seqs_ribonanza[37858:]
txt_bprna = open(Path("bpRNA_1m/dbnFiles/allDbn.dbn"), "r").read()
seqs_bprna = pd.Series(txt_bprna.split("\n")[1::3])

seqs_ribonanza = clean_sequences(seqs_ribonanza)
seqs_bprna = clean_sequences(seqs_bprna)

# struct1 = pd.read_csv(Path("stanford_ribonenza_data/supplementary_silico_predictions/GPN15k_silico_predictions.csv"))
# struct2 = pd.read_csv(Path("stanford_ribonenza_data/supplementary_silico_predictions/PK50_silico_predictions.csv"))
# struct3 = pd.read_csv(Path("stanford_ribonenza_data/supplementary_silico_predictions/PK90_silico_predictions.csv"))
# struct4 = pd.read_csv(Path("stanford_ribonenza_data/supplementary_silico_predictions/R1_silico_predictions.csv"))

res = []
for i, x in enumerate(seqs_ribonanza.values):
    if i % 1000 == 0:
        print(f"{round(i/len(seqs_ribonanza)*100, 1)}%")
    res.append(seqs_bprna.str.contains(x).sum())
