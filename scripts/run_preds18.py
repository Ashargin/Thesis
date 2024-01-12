import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path

from src.predict import (
    divide_predict,
    rnasubopt_predict,
)
from src.utils import run_preds

run_preds(
    divide_predict,
    Path("resources/divide_oracle_1000_sub04_16S23S.csv"),
    in_filename="16S23S",
    use_structs=True,
    feed_structs_to_print_fscores=True,
    kwargs={
        "max_length": 1000,
        "cut_model": None,  # with motifs input format
        "predict_fnc": lambda x: rnasubopt_predict(x, delta=0.4),
    },
)
