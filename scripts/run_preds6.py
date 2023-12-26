import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path

from src.predict import (
    mxfold2_predict,
)
from src.utils import run_preds

run_preds(
    mxfold2_predict,
    Path("resources/mxfold2.csv"),
    in_filename="test_sequencewise",
    allow_errors=True,
)
