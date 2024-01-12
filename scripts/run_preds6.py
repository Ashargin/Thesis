import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path

from src.predict import (
    divide_predict,
    ensemble_predict,
)
from src.utils import run_preds

run_preds(
    divide_predict,
    Path("resources/divide_oracle_1000_ens_sequencewise.csv"),
    in_filename="test_sequencewise",
    use_structs=True,
    kwargs={
        "max_length": 1000,
        "cut_model": None,  # with motifs input format
        "predict_fnc": ensemble_predict,
    },
)
