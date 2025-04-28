import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

source_file_path = Path(r"C:\Work\Thesis\bpRNA_1m\dbnFiles\allDbn.dbn")
txt = open(source_file_path, "r").read()
lines = txt.strip().split("\n")

names, seqs, structs = lines[::3], lines[1::3], lines[2::3]
lengths = [len(x) for x in seqs]

plt.figure(figsize=(7, 4))
sns.histplot(lengths, binwidth=50)
plt.xlim([0, max(lengths)])
plt.xlabel("Sequence length (nt)", fontsize=16)
plt.ylabel("Number of sequences", fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=14)
plt.show()
