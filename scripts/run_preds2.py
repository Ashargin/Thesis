import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path
from tensorflow import keras

from src.predict import (
    divide_predict,
    mxfold2_predict,
)
from src.models.loss import inv_exp_distance_to_cut_loss
from src.utils import run_preds

model = keras.models.load_model(
    Path("resources/models/CNN1D_familywise_85"), compile=False
)
model.compile(
    optimizer="adam",
    loss=inv_exp_distance_to_cut_loss,
    metrics=["accuracy"],
    run_eagerly=True,
)

run_preds(
    divide_predict,
    Path("resources/divide_with_mxfold2_familywise_85.csv"),
    in_filename="test_familywise_85",
    kwargs={
        "max_length": 1000,
        "cut_model": model,  # with motifs input format
        "predict_fnc": mxfold2_predict,
    },
)
