import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path

from src.predict import (
    divide_predict,
    rnafold_predict,
)
from src.utils import run_preds

run_preds(
    divide_predict,
    Path("resources/divide_oracle_1000_rnaf_familywise_80.csv"),
    in_filename="test_familywise_80",
    use_structs=True,  # Oracle
    kwargs={
        "max_length": 1000,
        "cut_model": None,
        "predict_fnc": rnafold_predict,
    },
)
