import os
import sys

sys.path.append(os.getcwd())

import re
from pathlib import Path
from tensorflow import keras

from src.predict import (
    divide_predict,
    mxfold2_predict,
)
from src.models.loss import inv_exp_distance_to_cut_loss
from src.utils import run_preds

# Settings
model_filename = "CNN1D_sequencewise_200motifs512dilINV"
max_length = 200
fuse_to = None

# Load model
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
if fuse_to is not None:
    model_name += f"{fuse_to}fuse"

max_motifs = (
    293
    if "motifs" not in model_filename
    else int(re.search("([0-9]*)motifs", model_filename).group(1))
)

# Run cutting metrics
run_preds(
    divide_predict,
    Path(f"resources/divide_{model_name}_{max_length}_sequencewise.csv"),
    in_filename="test_sequencewise",
    kwargs={
        "max_length": max_length,
        "cut_model": model,
        "predict_fnc": None,
        "max_motifs": max_motifs,
        "fuse_to": fuse_to,
    },
    evaluate_cutting_model=True,
)

# Run whole prediction
run_preds(
    divide_predict,
    Path(f"resources/divide_{model_name}_{max_length}_mx_sequencewise.csv"),
    in_filename="test_sequencewise",
    kwargs={
        "max_length": max_length,
        "cut_model": model,
        "predict_fnc": mxfold2_predict,
        "max_motifs": max_motifs,
        "fuse_to": fuse_to,
    },
    evaluate_cutting_model=False,
)
