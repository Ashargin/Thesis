from pathlib import Path
import os
import sys

sys.path.append(os.getcwd())

from src.predict import (
    mxfold2_predict,
    linearfold_predict,
    ufold_predict,
    probknot_predict,
    divide_predict,
    divide_get_cuts,
    linearfold_get_cuts,
)
from src.utils import run_preds

run_preds(
    probknot_predict,
    Path("resources/probknot.csv"),
)
