import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path

from src.predict import (
    divide_predict,
    linearfold_predict,
)
from src.utils import run_preds

run_preds(
    divide_predict,
    Path("resources/divide_oracle_1000_lf_16S.csv"),
    in_filename="16S",
    use_structs=True,  # Oracle
    kwargs={
        "max_length": 1000,
        "cut_model": None,  # with motifs input format
        "predict_fnc": linearfold_predict,
    },
)
