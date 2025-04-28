import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

df_sequencewise = pd.read_csv("resources/results/sequencewise_pk.csv")
df_familywise = pd.read_csv("resources/results/familywise_pk.csv")
df_validation = pd.read_csv("resources/results/validation.csv")
df_sequencewise["Test dataset"] = "Sequence-wise"
df_familywise["Test dataset"] = "Family-wise"
df_validation["Test dataset"] = "Validation"
data = pd.concat([df_sequencewise, df_familywise]).reset_index(drop=True)
data = data[data.cut_compression > 0]
data = data[data.model.apply(lambda x: "KnotFold" in x)]  # or "LinearFold" in x)]


# def plot(model):
#     df = data.copy()
#     df = df[df.model.apply(lambda x: model in x)]
#     df["weight"] = 1.
#     df.loc[df["Test dataset"] == "Family-wise", "weight"] = df[df["Test dataset"] == "Sequence-wise"].shape[0] / df[df["Test dataset"] == "Family-wise"].shape[0]
#     sns.kdeplot(df, x="cut_compression", y="cut_break_rate", hue="Test dataset", weights="weight", common_norm=False, levels=[0.2, 0.4, 0.6, 0.8])
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.xlabel("Compression rate")
#     plt.ylabel("Break rate")
#     plt.show()


# data = data[(data.length >= 400) & (data.length < 1000)]
# plot("CNN (400)")

## Sequence-wise
df_sequencewise = df_sequencewise[
    df_sequencewise.model.apply(lambda x: " CNN400 " in x and "KnotFold" in x)
]
df_sequencewise = df_sequencewise[
    df_sequencewise.length >= 1000
]  # RNAs longer than 1000 nc.
data = df_sequencewise.groupby("model")[
    ["cut_compression", "cut_break_rate", "fscore"]
].mean()
data["name"] = data.index
data["Maximum fragment length"] = data.name.apply(
    lambda x: int(re.search("\(([0-9]*)\)", x).group(1))
)
# data["predmodel"] = data.name.apply(lambda x: re.search("\+ (.*)", x).group(1))
data.drop("name", axis=1, inplace=True)
data.sort_values(["Maximum fragment length"], inplace=True)
dfs = [
    data.loc[:, ["Maximum fragment length", "cut_compression"]],
    data.loc[:, ["Maximum fragment length", "cut_break_rate"]],
    data.loc[:, ["Maximum fragment length", "fscore"]],
]
dfs[0].rename(columns={"cut_compression": "value"}, inplace=True)
dfs[1].rename(columns={"cut_break_rate": "value"}, inplace=True)
dfs[2].rename(columns={"fscore": "value"}, inplace=True)
dfs[0]["Metric"] = "DivideFold compression rate"
dfs[1]["Metric"] = "DivideFold break rate"
dfs[2]["Metric"] = "F-score : DivideFold + KnotFold"
new = pd.concat(dfs)
fig, ax = plt.subplots(figsize=(8.5, 4))
sns.lineplot(
    new,
    x="Maximum fragment length",
    y="value",
    hue="Metric",
    style="Metric",
    markers=True,
    dashes=False,
    ax=ax,
)
ax.tick_params(axis="both", which="major", labelsize=11)
ax.legend(title="Metric", prop={"size": 11}, title_fontsize=11)
ax.set_xlabel("Maximum fragment length", fontsize=16)
plt.xlim([200, 1200])
plt.ylim([0.0, 1.0])
plt.ylabel("")
plt.show()

## Family-wise
df_familywise = df_familywise[
    df_familywise.model.apply(lambda x: " CNN400 " in x and "IPknot" in x)
]
df_familywise = df_familywise[df_familywise.length >= 500]  # RNAs longer than 500 nc.
data = df_familywise.groupby("model")[
    ["cut_compression", "cut_break_rate", "fscore"]
].mean()
data["name"] = data.index
data["Maximum fragment length"] = data.name.apply(
    lambda x: int(re.search("\(([0-9]*)\)", x).group(1))
)
# data["predmodel"] = data.name.apply(lambda x: re.search("\+ (.*)", x).group(1))
data.drop("name", axis=1, inplace=True)
data.sort_values(["Maximum fragment length"], inplace=True)
dfs = [
    data.loc[:, ["Maximum fragment length", "cut_compression"]],
    data.loc[:, ["Maximum fragment length", "cut_break_rate"]],
    data.loc[:, ["Maximum fragment length", "fscore"]],
]
dfs[0].rename(columns={"cut_compression": "value"}, inplace=True)
dfs[1].rename(columns={"cut_break_rate": "value"}, inplace=True)
dfs[2].rename(columns={"fscore": "value"}, inplace=True)
dfs[0]["Metric"] = "DivideFold compression rate"
dfs[1]["Metric"] = "DivideFold break rate"
dfs[2]["Metric"] = "F-score : DivideFold + IPknot"
new = pd.concat(dfs)
fig, ax = plt.subplots(figsize=(8.5, 4))
sns.lineplot(
    new,
    x="Maximum fragment length",
    y="value",
    hue="Metric",
    style="Metric",
    markers=True,
    dashes=False,
    ax=ax,
)
ax.tick_params(axis="both", which="major", labelsize=11)
ax.legend(title="Metric", prop={"size": 11}, title_fontsize=11)
ax.set_xlabel("Maximum fragment length", fontsize=16)
plt.xlim([100, 500])
plt.ylim([0.0, 1.0])
plt.ylabel("")
plt.show()

## Validation
df_validation = df_validation[
    df_validation.model.apply(
        lambda x: " CNN1600EVOAUGINCRANGE " in x and "KnotFold" in x
    )
]
df_validation = df_validation[df_validation.length >= 1200]  # RNAs longer than 1000 nc.
data = df_validation.groupby("model")[
    ["cut_compression", "cut_break_rate", "fscore_nopk"]
].mean()
data["name"] = data.index
data["Maximum fragment length"] = data.name.apply(
    lambda x: int(re.search("\(([0-9]*)\)", x).group(1))
)
data = data[~data["Maximum fragment length"].isin([100, 300, 500])]
# data["predmodel"] = data.name.apply(lambda x: re.search("\+ (.*)", x).group(1))
data.drop("name", axis=1, inplace=True)
data.sort_values(["Maximum fragment length"], inplace=True)
dfs = [
    data.loc[:, ["Maximum fragment length", "cut_compression"]],
    data.loc[:, ["Maximum fragment length", "cut_break_rate"]],
    data.loc[:, ["Maximum fragment length", "fscore_nopk"]],
]
dfs[0].rename(columns={"cut_compression": "value"}, inplace=True)
dfs[1].rename(columns={"cut_break_rate": "value"}, inplace=True)
dfs[2].rename(columns={"fscore_nopk": "value"}, inplace=True)
dfs[0]["Metric"] = "DivideFold compression rate"
dfs[1]["Metric"] = "DivideFold break rate"
dfs[2]["Metric"] = "F-score : DivideFold + KnotFold"
new = pd.concat(dfs)
fig, ax = plt.subplots(figsize=(8.5, 4))
sns.lineplot(
    new,
    x="Maximum fragment length",
    y="value",
    hue="Metric",
    style="Metric",
    markers=True,
    dashes=False,
    ax=ax,
)
ax.tick_params(axis="both", which="major", labelsize=14)
ax.legend(title="Metric", prop={"size": 14}, title_fontsize=14, ncol=2)
ax.set_xlabel("Maximum fragment length (nt)", fontsize=16)
plt.xlim([200, 1200])
plt.ylim([0.0, 1.0])
plt.ylabel("")
plt.show()
