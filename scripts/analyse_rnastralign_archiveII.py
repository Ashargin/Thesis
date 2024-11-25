from pathlib import Path
import os

path = Path(r"C:\Users\loico\Downloads\RNAStrAlign_bpseq\RNAStrAlign_bpseq")

list1 = os.listdir(path)
if list1[0] == ".DS_Store":
    list1 = list1[1:]

lens = []
types = []
for f1 in list1:
    print()
    print(f"DIR1 {f1}")
    list2 = os.listdir(path / f1)

    if list2[0] == ".DS_Store":
        list2 = list2[1:]

        for f2 in list2:
            print(f"DIR2 {f2}")
            for f in os.listdir(path / f1 / f2):
                filepath = path / f1 / f2 / f
                assert f.endswith(".bpseq")
                txt = open(filepath, "r").read()
                lines = txt.strip().split("\n")
                ok = True
                for i, l in enumerate(lines):
                    if int(l.split(" ")[0]) != i + 1:
                        ok = False
                if ok:
                    lens.append(len(lines))
                    types.append(f1 + "_" + f2)

    else:
        for f in list2:
            filepath = path / f1 / f
            assert f.endswith(".bpseq")
            txt = open(filepath, "r").read()
            lines = txt.strip().split("\n")
            ok = True
            for i, l in enumerate(lines):
                if int(l.split(" ")[0]) != i + 1:
                    ok = False
            if ok:
                lens.append(len(lines))
                types.append(f1)

import pandas as pd

df = pd.DataFrame({"type": types, "length": lens})
