import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path
from tensorflow import keras

from src.predict import (
    dividefold_predict,
    mxfold2_predict,
    rnafold_predict,
    linearfold_predict,
    knotfold_predict,
    ipknot_predict,
    pkiss_predict,
    probknot_predict,
)
from src.models.loss import inv_exp_distance_to_cut_loss
from src.utils import run_preds

# Settings
global_predict_fnc = dividefold_predict
model_filename = "CNN1D"
predict_fnc = linearfold_predict
evaluate_cutting_model = False
max_length = 400

# Load model
model = None
model_name = ""
if global_predict_fnc.__name__ == "dividefold_predict":
    model_name = "oracle"
    if model_filename != "oracle":
        model = keras.models.load_model(
            Path(f"resources/models/{model_filename}"), compile=False
        )
        model.compile(
            optimizer="adam",
            loss=inv_exp_distance_to_cut_loss,
            run_eagerly=True,
        )

        model_name = (
            model_filename.replace("_", "")
            .replace("CNN1D", "cnn")
            .replace("MLP", "mlp")
            .replace("BiLSTM", "bilstm")
        )

global_model_name = global_predict_fnc.__name__.replace("_predict", "")
model_name = (
    "_" + model_name if global_predict_fnc.__name__ == "dividefold_predict" else ""
)
max_length_name = (
    f"_{'meta' if max_length is None else max_length}"
    if global_predict_fnc.__name__ == "dividefold_predict"
    else ""
)
predict_name = ""
if global_predict_fnc.__name__ == "dividefold_predict" and not evaluate_cutting_model:
    predict_name = "_" + predict_fnc.__name__.replace("_predict", "").replace(
        "mxfold2", "mx"
    ).replace("linearfold", "lf").replace("rnafold", "rnaf").replace(
        "knotfold", "kf"
    ).replace(
        "ipknot", "ipk"
    ).replace(
        "pkiss", "pks"
    ).replace(
        "probknot", "pbk"
    )
kwargs = (
    {"cut_model": model, "predict_fnc": predict_fnc, "max_length": max_length}
    if global_predict_fnc.__name__ == "dividefold_predict"
    else {}
)

# Run cutting metrics
for dataset in [
    "test_familywise"
]:  # "16S23S", "curated_lncRNAs", "test_familywise", "test_sequencewise"]:
    dataset_name = dataset.replace("test_", "").replace("_lncRNAs", "")
    run_preds(
        global_predict_fnc,
        Path(
            f"resources/{global_model_name}{model_name}{max_length_name}{predict_name}_{dataset_name}.csv"
        ),
        in_filename=dataset,
        allow_errors=global_predict_fnc.__name__
        in ["mxfold2_predict", "knotfold_predict", "ipknot_predict", "pkiss_predict"],
        use_structs=model_filename == "oracle",
        kwargs=kwargs,
        evaluate_cutting_model=evaluate_cutting_model,
    )
