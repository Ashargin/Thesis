import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path

from src.predict import (
    probknot_predict,
)
from src.utils import run_preds

run_preds(
    probknot_predict,
    Path("resources/probknot.csv"),
    in_filename="test_sequencewise",
)
