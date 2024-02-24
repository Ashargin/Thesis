import os
import sys

sys.path.append(os.getcwd())

import re
from pathlib import Path
from tensorflow import keras

from src.predict import (
    divide_predict,
)
from src.models.loss import inv_exp_distance_to_cut_loss
from src.utils import run_preds

model_filename = "CNN1D_sequencewise_200motifs512dilINV"
max_motifs = (
    293
    if "motifs" not in model_filename
    else int(re.search("([0-9]*)motifs", model_filename).group(1))
)
model = keras.models.load_model(
    Path(f"resources/models/{model_filename}"), compile=False
)
model.compile(
    optimizer="adam",
    loss=inv_exp_distance_to_cut_loss,
    metrics=["accuracy"],
    run_eagerly=True,
)

model_name = (
    model_filename.replace("_", "")
    .replace("sequencewise", "")
    .replace("CNN1D", "cnn")
    .replace("MLP", "mlp")
    .replace("BiLSTM", "bilstm")
    .replace("EPOCH10", "")
)
run_preds(
    divide_predict,
    Path(f"resources/divide_{model_name}200gap_1000_sequencewise.csv"),
    in_filename="test_sequencewise",
    kwargs={
        "max_length": 1000,
        "cut_model": model,
        "predict_fnc": None,
        "max_motifs": max_motifs,
    },
    evaluate_cutting_model=True,
)
