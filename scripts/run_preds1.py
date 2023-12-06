from pathlib import Path
import os
import sys

sys.path.append(os.getcwd())

from src.predict import (
    mxfold2_predict,
    linearfold_predict,
    ufold_predict,
    rnafold_predict,
    divide_predict,
    divide_get_cuts,
    linearfold_get_cuts,
)
from src.utils import run_preds

run_preds(
    divide_predict,
    Path("resources/divide_with_rnafold.csv"),
    kwargs={
        "max_length": 1000,
        "cut_fnc": divide_get_cuts,  # with motifs input format
        "predict_fnc": rnafold_predict,
    },
)
