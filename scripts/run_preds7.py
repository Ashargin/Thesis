import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path

from src.predict import (
    linearfold_predict,
)
from src.utils import run_preds

run_preds(
    linearfold_predict,
    Path("resources/linearfold.csv"),
    in_filename="test_sequencewise",
)
