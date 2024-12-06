import os
import pickle
import numpy as np
import pandas as pd
import keras

# import torch
# from transformers import AutoTokenizer, AutoModel
import scipy.signal
from pathlib import Path

from src.utils import format_data, apply_mutation, evoaug_augment
from src.models.mlp import MLP
from src.models.cnn_1d import CNN1D
from src.models.bilstm import BiLSTM
from src.predict import oracle_get_cuts

# from src.utils import seq2kmer

MAX_MOTIFS = 200
MAX_DIL = 256
DATA_AUGMENT_MUTATION = True
MIN_LEN = 1600
AUGMENT_STRUCTS = False
AUGMENT_TYPE = "EVOAUG"

# Load model
model = CNN1D(input_shape=(None, MAX_MOTIFS + 4), max_dil=MAX_DIL)
# model = MLP(input_shape=(None, MAX_MOTIFS + 4))
# model = BiLSTM(input_shape=(None, MAX_MOTIFS + 4))
pretrained_model = keras.models.load_model(r"resources/models/CNN1D.keras")
model.set_weights(pretrained_model.weights)
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    run_eagerly=True,
)


# Dataloader
def motif_data_generator(
    path_in,
    max_motifs=MAX_MOTIFS,
    min_len=MIN_LEN,
    max_len=None,
    from_cache=False,
    data_augment_mutation=DATA_AUGMENT_MUTATION,
    loss_lbda=0.5,
):
    files = None
    df_in = None
    if from_cache:
        files = os.listdir(path_in)
        np.random.shuffle(files)
    else:
        path_df_in = path_in.parent / (path_in.name + ".csv")
        df_in = pd.read_csv(path_df_in, index_col=0)
        df_in = df_in.sample(frac=1.0)
        if min_len is not None:
            df_in = df_in[df_in.seq.apply(len) > min_len]
        if max_len is not None:
            df_in = df_in[df_in.seq.apply(len) <= max_len]
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
            row = df_in.iloc[i % df_in.shape[0]]
            seq, struct, cuts = row.seq, row.struct, row.cuts

            if data_augment_mutation:
                new_seq, new_struct = (
                    apply_mutation(
                        seq,
                        struct,
                        mutation_proba=0.1 * np.random.random(),
                        struct_deletion_proba=0.1 * np.random.random()
                        if AUGMENT_STRUCTS
                        else 0.0,
                    )
                    if AUGMENT_TYPE != "EVOAUG"
                    else evoaug_augment(seq, struct)
                )
                seq = new_seq
                if new_struct != struct:
                    struct = new_struct
                    cuts, outer = oracle_get_cuts(struct)
                    cuts = str(cuts).replace(",", "")

            seq_mat = format_data(seq, max_motifs=max_motifs)
            cuts_mat = np.array([float(c) for c in cuts[1:-1].split(" ")])

            # Inverse exponential distance to cut points loss
            loss_array = np.abs(
                cuts_mat.reshape((1, -1)) - np.arange(len(seq)).reshape((-1, 1))
            ).min(axis=1)
            loss_array = np.exp(-loss_lbda * loss_array)

        i += 1
        if (max_len is not None and len(seq) > max_len) or (
            min_len is not None and len(seq) <= min_len
        ):
            continue

        assert seq_mat.shape == (len(seq), max_motifs + 4)
        assert loss_array.shape == (len(seq),)

        yield seq_mat.reshape((1, len(seq), max_motifs + 4)), loss_array.reshape(
            (1, len(seq))
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
train_path = Path("resources/data_splits/train")
val_path = Path("resources/data_splits/validation_sequencewise")
train_gen = motif_data_generator(train_path)
val_gen = motif_data_generator(val_path, data_augment_mutation=False)
histories = []
losses = []
i = 0
while True:
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=3780,  # 37796|27768301 / 9138|13545894 / 332|782530
        # 10-50 (1-5)    / 5-25 (2-10)   / 3-15 (35-175)
        epochs=1,
        validation_steps=333,  # 3334 / 761 / 70
    )
    histories.append(history.history)
    this_loss = round(100000 * np.mean(history.history["val_loss"]), 2)

    must_save = (not losses) or (this_loss < min(losses))
    if must_save:
        model.save(Path(f"resources/models/CNN1D_{MIN_LEN}{AUGMENT_TYPE}.keras"))
    losses.append(this_loss)
    print(f"Epoch {i+1}:")
    print(losses)
    print("SAVED" if must_save else "DISCARDED")
    i += 1

# import matplotlib.pyplot as plt

#
# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# X = np.arange(len(loss))
# plt.plot(X, loss, label="Train loss")
# plt.plot(X, val_loss, label="Validation loss")
# plt.legend()
# plt.xlim(([0, 100]))
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training loss curve (sequence-wise train / val split)")
# plt.savefig(
#     rf"resources/png/training_curve_cnn_sequencewise_{MAX_MOTIFS}motifs{MAX_DIL}dilINV{'_augmented' if DATA_AUGMENT_MUTATION else ''}.png"
# )
# plt.show()
#
# my_model = keras.models.load_model(Path("resources/models/CNN1D"), compile=False)
# my_model.compile(
#     optimizer="adam",
#     loss=inv_exp_distance_to_cut_loss,
#     run_eagerly=True,
# )
# test_datagen = motif_data_generator(Path("resources/data_splits/test_sequencewise"))
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
#     peaks = scipy.signal.find_peaks(preds, height=0.28, distance=12)[0]
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
