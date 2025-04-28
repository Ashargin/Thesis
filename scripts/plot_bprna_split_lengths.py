import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

train_split_path = Path(r"C:\Work\Thesis\resources\data_splits\train.csv")
df = pd.read_csv(train_split_path)
df["length"] = df.seq.apply(len)

train_file_path = Path(r"C:\Work\Thesis\resources\data_structures\train.dbn")
txt = open(train_file_path, "r").read()
lines = txt.strip().split("\n")
names, seqs, structs = lines[::3], lines[1::3], lines[2::3]
names = [n.split("#Name: ")[1] for n in names]
lengths = [len(x) for x in seqs]
df_orig = pd.DataFrame(
    {"rna_name": names, "seq": seqs, "struct": structs, "length": lengths}
)
df_orig = df_orig[df_orig.length >= df.length.min()]

data = pd.DataFrame(
    {
        "length": df_orig.length.tolist() + df.length.tolist(),
        "source": ["Original sequences in Train"] * df_orig.shape[0]
        + ["Total training samples"] * df.shape[0],
    }
)
data.length -= 1

plt.figure(figsize=(7, 4))
ax = sns.histplot(data, x="length", hue="source", binwidth=50)
plt.xlim([100, data.length.max()])
plt.ylim([0, 35000])
xticks = ax.get_xticklabels()
positions = [xt.get_position()[0] for xt in xticks]
positions[0] = 100
texts = [xt.get_text() for xt in xticks]
texts[0] = "100"
plt.xticks(positions, texts)
yticks = ax.get_yticklabels()
positions = [yt.get_position()[1] for yt in yticks]
positions = positions[::2]
texts = [yt.get_text() for yt in yticks]
texts = texts[::2]
plt.yticks(positions, texts)
plt.xlabel("Sequence length (nt)", fontsize=16)
plt.ylabel("Number of sequences", fontsize=16)
plt.tick_params(axis="both", which="major", labelsize=14)
plt.show()

plt.figure(figsize=(7, 4))
ax = sns.boxplot(data, x="length", hue="source")
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
plt.legend(
    loc="lower left",
    title="",
    prop={"size": 14},
    title_fontsize=14,
    ncol=4,
    bbox_to_anchor=(0.0, -0.3),
)
plt.show()
