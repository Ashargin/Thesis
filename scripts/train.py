import os
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras

# import torch
# from transformers import AutoTokenizer, AutoModel
from scipy import signal
from pathlib import Path

from src.utils import format_data, apply_mutation
from src.models.mlp import MLP
from src.models.cnn_1d import CNN1D
from src.models.bilstm import BiLSTM
from src.models.loss import inv_exp_distance_to_cut_loss

# from src.utils import seq2kmer

MAX_MOTIFS = 200
MAX_DIL = 512
DATA_AUGMENT_MUTATION = True
FROM_CACHE = not DATA_AUGMENT_MUTATION

# Load model
model = CNN1D(input_shape=(None, MAX_MOTIFS + 4), max_dil=MAX_DIL)
# model = BiLSTM(input_shape=(None, MAX_MOTIFS + 4))
model.compile(
    optimizer="adam",
    loss=inv_exp_distance_to_cut_loss,
    metrics=["accuracy"],
    run_eagerly=True,
)


# Dataloader
def motif_cache_data_generator(
    path_in,
    max_motifs=MAX_MOTIFS,
    max_len=None,
    from_cache=FROM_CACHE,
    data_augment_mutation=DATA_AUGMENT_MUTATION,
):
    files = os.listdir(path_in)
    np.random.shuffle(files)
    if not from_cache:
        path_df_in = Path("resources/data_splits") / (path_in.name + ".csv")
        df_in = pd.read_csv(path_df_in, index_col=0)
        idx = [int(f.split(".pkl")[0]) for f in files]
        df_in = df_in.loc[idx, :]
        assert df_in.isna().sum().sum() == 0

    df_motifs = pd.read_csv(Path("resources/motif_seqs.csv"), index_col=0)
    df_motifs = df_motifs[df_motifs.time < 0.012].reset_index(drop=True)
    max_motifs = df_motifs.shape[0] if max_motifs is None else max_motifs
    motif_used_index = (
        df_motifs.sort_index().sort_values("time").index[:max_motifs].sort_values()
    )
    used_index = [0, 1, 2, 3] + (motif_used_index + 4).to_list()  # add one-hot

    i = 0
    while True:
        seq_mat, cuts_mat, outer = None, None, None
        if from_cache:
            if data_augment_mutation:
                raise Warning("Mutation data augmentation is unavailable from cache.")
            with open(path_in / files[i % len(files)], "rb") as infile:
                seq_mat, cuts_mat, outer = pickle.load(infile)

            seq_mat = seq_mat.toarray()
            cuts_mat = cuts_mat.toarray()
            # outer = outer.toarray()

            seq_mat = seq_mat[:, used_index]  # keep top max_motifs motifs
            cuts_mat = np.where(cuts_mat.ravel() == 1)[0].astype(float)  # cut indices

        else:
            row = df_in.iloc[i % len(files)]
            seq, struct, cuts = row.seq, row.struct, row.cuts

            if data_augment_mutation:
                seq, struct = apply_mutation(
                    seq, struct, mutation_proba=np.random.random()
                )

            seq_mat = format_data(seq, max_motifs=max_motifs)
            cuts_mat = np.array([float(c) for c in cuts[1:-1].split(" ")])

        i += 1
        if max_len is not None and seq_mat.shape[0] > max_len:
            continue

        yield seq_mat.reshape((1, seq_mat.shape[0], max_motifs + 4)), cuts_mat.reshape(
            (1, cuts_mat.shape[0])
        )


# def dnabert_data_generator(csv_path, max_len=None):
#     dnabert_tokenizer = AutoTokenizer.from_pretrained(
#         "zhihan1996/DNA_bert_6", trust_remote_code=True
#     )
#     dnabert_encoder = AutoModel.from_pretrained(
#         "zhihan1996/DNA_bert_6", trust_remote_code=True
#     )
#
#     df = pd.read_csv(csv_path, index_col=0)
#     df = df.sample(frac=1.0)
#
#     seq_mat, cuts_mat, outer = None, None, None
#     i = 0
#     while True:
#         seq = df.iloc[i].seq
#         cuts = [int(x) for x in df.iloc[i].cuts[1:-1].split()]
#
#         i += 1
#         if max_len is not None and len(seq) > max_len:
#             continue
#
#         tokenized = dnabert_tokenizer(
#             seq2kmer(seq.replace("U", "T"), k=6),
#             padding="longest",
#             pad_to_multiple_of=512,
#         )
#         encoded = dnabert_encoder(
#             torch.tensor([tokenized["input_ids"]]).view(-1, 512),
#             torch.tensor([tokenized["attention_mask"]]).view(-1, 512),
#         )
#         seq_mat, pooled_seq_mat = encoded[0], encoded[1]
#
#         tokens_len = len(seq) - 3
#         seq_mat = np.vstack(seq_mat.detach().numpy())[:tokens_len]
#         seq_mat = np.vstack(
#             [seq_mat[0], seq_mat[0], seq_mat, seq_mat[-1]]
#         )  ###### fix size ?
#         # pooled_seq_mat = np.mean(pooled_seq_mat.detach().numpy(), axis=0)
#         cuts_mat = np.array(cuts)
#
#         # explore LLMs / generatives / T5
#
#         yield seq_mat.reshape((1, len(seq), 768)), cuts_mat.reshape((1, len(cuts)))


# Fit model
train_path = Path("resources/data_splits/train_sequencewise")
test_path = Path("resources/data_splits/test_sequencewise")
history = model.fit(
    motif_cache_data_generator(train_path),
    validation_data=motif_cache_data_generator(test_path),
    steps_per_epoch=1267,
    epochs=100,
    validation_steps=305,
)

model.save(
    Path(
        f"resources/models/CNN1D_sequencewise_{MAX_MOTIFS}motifs{MAX_DIL}dilINV{'_augmented' if DATA_AUGMENT_MUTATION else ''}"
    )
)

import matplotlib.pyplot as plt

loss = history.history["loss"]
val_loss = history.history["val_loss"]
X = np.arange(len(loss))
plt.plot(X, loss, label="Train loss")
plt.plot(X, val_loss, label="Validation loss")
plt.legend()
plt.xlim(([0, 100]))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss curve (sequence-wise train / test split)")
plt.savefig(
    rf"resources/png/training_curve_cnn_sequencewise_{MAX_MOTIFS}motifs{MAX_DIL}dilINV{'_augmented' if DATA_AUGMENT_MUTATION else ''}.png"
)
plt.show()

# my_model = keras.models.load_model(Path("resources/models/model_motifs"), compile=False)
# my_model.compile(
#     optimizer="adam",
#     loss=inv_exp_distance_to_cut_loss,
#     metrics=["accuracy"],
#     run_eagerly=True,
# )
# test_datagen = motif_cache_data_generator(Path("resources/data/temptest"))
#
#
# def plot_cut_probabilities():
#     seq_mat, cuts_mat = next(test_datagen)
#     preds = my_model(seq_mat).numpy().ravel()
#     cuts_mat = cuts_mat.ravel().astype(int)
#
#     X = np.arange(len(preds)) + 1
#     for i, x in enumerate(X[cuts_mat]):
#         plt.plot(
#             [x, x],
#             [0, 1],
#             color="black",
#             linewidth=1.5,
#             label="True cut points" if i == 0 else "",
#         )
#     plt.plot(X, preds, color="tab:orange", label="Predicted probabilities to cut")
#
#     peaks = signal.find_peaks(preds, height=0.28, distance=12)[0]
#     plt.plot(X[peaks], preds[peaks], "o", color="tab:blue", label="Selected cut points")
#
#     plt.xlim([X[0], X[-1]])
#     plt.ylim([0, 1])
#
#     plt.title(
#         "Predicted cutting probabilities and selected cut points\ncompared to true cut points"
#     )
#     plt.legend()
#     plt.show()
