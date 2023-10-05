from predict import (
    mxfold2_predict,
    linearfold_predict,
    ufold_predict,
    divide_predict,
    divide_get_cuts,
    linearfold_get_cuts,
)
from utils import run_preds

run_preds(
    divide_predict,
    "resources/divide_linearfoldcuts_1step_mx_preds.csv",
    kwargs={
        "max_steps": 1,
        "cut_fnc": linearfold_get_cuts,
        "predict_fnc": mxfold2_predict,
    },
    compute_frac=0.2,
)
