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
model_filename = "CNN1D_sequencewise_200motifs512dilINV_augmented"
max_length = None
min_steps = None
fuse_to = 400

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

max_length_name = ""
if min_steps is not None:
    max_length_name += f"{min_steps}mins"
if max_length is not None:
    max_length_name += str(max_length)
if min_steps is None and max_length is None:
    max_length_name = "meta"

max_motifs = (
    293
    if "motifs" not in model_filename
    else int(re.search("([0-9]*)motifs", model_filename).group(1))
)

# Run cutting metrics
run_preds(
    divide_predict,
    Path(f"resources/divide_{model_name}_{max_length_name}_sequencewise.csv"),
    in_filename="test_sequencewise",
    kwargs={
        "max_length": max_length,
        "min_steps": min_steps,
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
    Path(f"resources/divide_{model_name}_{max_length_name}_mx_sequencewise.csv"),
    in_filename="test_sequencewise",
    kwargs={
        "max_length": max_length,
        "min_steps": min_steps,
        "cut_model": model,
        "predict_fnc": mxfold2_predict,
        "max_motifs": max_motifs,
        "fuse_to": fuse_to,
    },
    evaluate_cutting_model=False,
)
