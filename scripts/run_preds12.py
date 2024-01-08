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
    Path("resources/rnafold_16S.csv"),
    in_filename="16S",
)
