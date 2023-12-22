import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path

from src.predict import (
    rnafold_predict,
)
from src.utils import run_preds

run_preds(
    rnafold_predict,
    Path("resources/rnafold.csv"),
    in_filename="test_sequencewise",
)
