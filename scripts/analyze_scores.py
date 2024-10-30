from pathlib import Path
import numpy as np
import pandas as pd
import re
import keras
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.utils import format_data, struct_to_pairs
from src.models.loss import inv_exp_distance_to_cut_loss
from src.predict import dividefold_predict

# Load scores
df = pd.read_csv("temp.csv")
# df = df[df.length > 2500].sort_values('fscore')

cut_model = keras.layers.TFSMLayer(
    Path("resources/models/CNN1D"),
    call_endpoint="serving_default",
)


def analyze_cutting(seq, struct, with_curve=True):
    n = len(seq)
    pairs_nopk = struct_to_pairs(re.sub("[^\(\)\.]", ".", struct))
    pairs_pk = struct_to_pairs(re.sub("[\(\)]", ".", struct))

    # Predict
    frags, _, _ = dividefold_predict(
        seq, max_length=1000, max_steps=1, min_steps=1, return_cuts=True
    )
    frags = [r[0] for r in frags]
    preds = None
    if with_curve:
        seq_mat = format_data(seq, max_motifs=200)[np.newaxis, :, :]
        preds = cut_model(seq_mat)
        preds = list(preds.values())[0].numpy().ravel()

    def draw_pairs(pairs, color):
        for i, j in enumerate(pairs):
            i += 1
            if i < j:
                x = np.array([i, (3 * i + j) / 4, (i + j) / 2, (i + 3 * j) / 4, j])
                y = np.array([0, 0.8 * (j - i) / n, (j - i) / n, 0.8 * (j - i) / n, 0])
                X_Y_Spline = make_interp_spline(x, y)
                X_ = np.linspace(i, j, 100)
                Y_ = X_Y_Spline(X_)
                plt.plot(X_, Y_, color=color, linewidth=0.5)

    # Draw
    draw_pairs(pairs_nopk, "silver")
    draw_pairs(pairs_pk, "firebrick")

    if with_curve:
        plt.plot(np.arange(n) + 1, preds, color="tab:orange")

    colors = list(mcolors.TABLEAU_COLORS.keys())
    np.random.shuffle(colors)
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    for p, frag in enumerate(frags):
        for i, j in frag:
            plt.plot([i - 0.5, j + 0.5], [-0.005, -0.005], color=colors[p], linewidth=5)

    plt.xlim([0, n])
    plt.ylim([-0.05, 1])
    plt.show()


seq = df.iloc[0].seq
struct = df.iloc[0].struct
analyze_cutting(seq, struct)
