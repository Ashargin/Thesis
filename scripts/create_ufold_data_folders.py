from pathlib import Path
import os
from src.utils import struct_to_pairs

path = Path(r"C:\Work\Thesis\resources\data_structures")

for f_in in os.listdir(path):
    dirname, _ = os.path.splitext(f_in)
    os.mkdir(path / dirname)
    lines = open(path / f_in, "r").read().strip().split("\n")
    names, seqs, structs = lines[::3], lines[1::3], lines[2::3]
    names = [n.split("#Name: ")[1] for n in names]
    for n, se, st in zip(names, seqs, structs):
        n = n.replace("/", "~")  # to allow text file creation
        pairs = struct_to_pairs(st)
        with open(path / dirname / (n + ".bpseq"), "w") as f_out:
            for i, p in enumerate(pairs):
                f_out.write(f"{i+1} {se[i]} {p}\n")
