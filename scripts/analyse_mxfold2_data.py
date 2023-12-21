import os
from pathlib import Path
import pandas as pd

root = Path("C:/Work/Thesis/mxfold2_data")
paths = {
    "ArchiveII": root / "archiveII",
    "TR0": root / "bpRNA_dataset-canonicals" / "TR0",
    "TS0": root / "bpRNA_dataset-canonicals" / "TS0",
    "VL0": root / "bpRNA_dataset-canonicals" / "VL0",
    "bpRNA-new": root / "bpRNAnew_dataset" / "bpRNAnew.nr500.canonicals",
    "TrainSetA": root / "TrainSetA",
    "TrainSetB": root / "TrainSetB",
    "TestSetA": root / "TestSetA",
    "TestSetB": root / "TestSetB",
}

lens = {}
for dataset, p in paths.items():
    lens[dataset] = []
    files = [filename for filename in os.listdir(p) if not filename.startswith("._")]
    for filename in files:
        with open(p / filename, "r") as f:
            txt = f.read()
            length = txt.count("\n")
            lens[dataset].append(length)

lens = {dataset: pd.Series(lens[dataset]) for dataset in lens}
datasets = list(lens.keys())
df = pd.DataFrame(
    [
        [
            lens[d].min(),
            lens[d].quantile(0.1).astype(int),
            lens[d].quantile(0.25).astype(int),
            lens[d].quantile(0.5).astype(int),
            lens[d].quantile(0.75).astype(int),
            lens[d].quantile(0.9).astype(int),
            lens[d].max(),
        ]
        for d in datasets
    ],
    index=datasets,
    columns=["min", "q10", "q25", "q50", "q75", "q90", "max"],
)

#            min  q10  q25  q50  q75  q90   max
# ArchiveII   28   76  109  120  315  380  1800
# TR0         33   69   80  105  152  234   498
# TS0         22   70   80  109  152  243   499
# VL0         33   69   80  106  151  230   497
# bpRNA-new   33   56   70   92  127  186   489
# TrainSetA   10   18   43  114  353  485   734
# TrainSetB   27   45   66  100  137  164   237
# TestSetA    10   16   30   74  293  479   768
# TestSetB    27   86  100  109  148  172   244
