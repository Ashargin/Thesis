from pathlib import Path

path = Path(r"C:\Work\Thesis\scripts")
main_funcs = ["dividefold_predict"] * 18 + [
    "mxfold2_predict",
    "rnafold_predict",
    "linearfold_predict",
    "knotfold_predict",
    "ipknot_predict",
    "pkiss_predict",
    "probknot_predict",
]
main_funcs = main_funcs * 2 + ["dividefold_predict"] * 5
models = (
    [
        '"BiLSTM_400"',
        '"MLP_400"',
        '"CNN1D_400"',
        '"CNN1D_1000"',
        '"CNN1D_1600"',
        '"CNN1D_1600EVOAUGCONFLICTMAX"',
        '"CNN1D_1600EVOAUGCONFLICTMIN"',
        '"CNN1D_1600EVOAUGINCLENGTH"',
        '"CNN1D_1600EVOAUGINCORDER"',
        '"CNN1D_1600EVOAUGINCRANGE"',
        '"CNN1D_1600EVOAUGOPTALL"',
        '"CNN1D_1600STRUCTAUG"',
    ]
    + ['"CNN1D_1600EVOAUGINCRANGE"'] * 6
    + [None] * 7
)
models = models * 2 + ['"CNN1D_1600EVOAUGINCRANGE"'] * 5
sub_funcs = (
    ["knotfold_predict"] * 12
    + [
        "mxfold2_predict",
        "rnafold_predict",
        "linearfold_predict",
        "ipknot_predict",
        "pkiss_predict",
        "probknot_predict",
    ]
    + [None] * 7
)
sub_funcs = sub_funcs * 2 + ["knotfold_predict"] * 5
max_lengths = [1000] * 25 + [500] * 25 + [200, 400, 600, 800, 1200]
lsts_datasets = (
    [["16S23S", "curated_lncRNAs", "validation", "test_sequencewise"]] * 25
    + [["test_familywise", "test_familywise15", "test_sequencewise"]] * 25
    + [
        [
            "16S23S",
            "curated_lncRNAs",
            "validation",
            "test_familywise",
            "test_familywise15",
            "test_sequencewise",
        ]
    ]
    * 4
    + [["16S23S", "curated_lncRNAs", "validation", "test_sequencewise"]]
)

for i in range(1, 56):
    with open(path / f"run_preds{i}.py", "w") as f:
        f.write(
            f"""import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path
import keras

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
from src.utils import run_preds

# Settings
global_predict_fnc = {main_funcs[i-1]}
model_filename = {models[i-1]}
predict_fnc = {sub_funcs[i-1]}
evaluate_cutting_model = True
max_length = {max_lengths[i-1]}
lst_datasets = {lsts_datasets[i-1]}
"""
            + """
# Load model
model = None
model_name = ""
if global_predict_fnc.__name__ == "dividefold_predict":
    model_name = "oracle"
    if model_filename != "oracle":
        model = keras.models.load_model(
            Path(f"resources/models/{model_filename}.keras")
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
for dataset in lst_datasets:
    dataset_name = dataset.replace("test_", "").replace("_lncRNAs", "")
    run_preds(
        global_predict_fnc,
        Path(
            f"resources/{global_model_name}{model_name}{max_length_name}{predict_name}_{dataset_name}.csv"
        ),
        in_filename=dataset,
        allow_errors=global_predict_fnc.__name__
        in ["mxfold2_predict", "knotfold_predict", "pkiss_predict"],
        use_structs=model_filename == "oracle",
        kwargs=kwargs,
        evaluate_cutting_model=evaluate_cutting_model,
    )
"""
        )
