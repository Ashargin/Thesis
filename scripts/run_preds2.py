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
    rnafold_predict,
    Path("resources/rnafold.csv"),
    allow_errors=True,
)
