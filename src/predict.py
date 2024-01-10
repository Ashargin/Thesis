import os
import sys
import random
from pathlib import Path
import time
import datetime
import re
import numpy as np
from scipy import signal
import pickle

import torch
from torch.utils.data import DataLoader
from tensorflow import keras

path_workdir = Path("..")
path_ufold = Path("../UFold")
path_linearfold = Path("../LinearFold")
path_rnapar = Path("../RNAPar")
path_rnastructure = Path("../RNAstructure/exe")
sys.path.append(os.path.abspath(path_workdir))
sys.path.append(os.path.abspath(path_ufold))
sys.path.append(os.path.abspath(path_rnastructure))
import mxfold2
from mxfold2.predict import Predict
from UFold.ufold_predict import main as main_ufold

from src.utils import format_data
from src.models.loss import inv_exp_distance_to_cut_loss

default_model = keras.models.load_model(
    Path("resources/models/CNN1D_sequencewise"), compile=False
)
default_model.compile(
    optimizer="adam",
    loss=inv_exp_distance_to_cut_loss,
    metrics=["accuracy"],
    run_eagerly=True,
)


class Args:
    # self.func=<function Predict.add_args.<locals>.<lambda> at 0x7f526585d4c0>
    # self.input='bpRNA_CRW_1.fasta'
    seed = 0
    gpu = 0
    param = "TrainSetAB.pth"
    result = None
    bpseq = None
    model = "MixC"
    max_helix_length = 30
    embed_size = 64
    num_filters = [64, 64, 64, 64, 64, 64, 64, 64]
    filter_size = [5, 3, 5, 3, 5, 3, 5, 3]
    pool_size = [1]
    dilation = 0
    num_lstm_layers = 2
    num_lstm_units = 32
    num_transformer_layers = 0
    num_transformer_hidden_units = 2048
    num_transformer_att = 8
    num_paired_filters = [64, 64, 64, 64, 64, 64, 64, 64]
    paired_filter_size = [5, 3, 5, 3, 5, 3, 5, 3]
    num_hidden_units = [32]
    dropout_rate = 0.5
    fc_dropout_rate = 0.5
    num_att = 8
    pair_join = "cat"
    no_split_lr = False


mxfold2_predictor = Predict()
args = Args()
conf = Path(mxfold2.__file__).parents[0] / "models/TrainSetAB.conf"

# seed
if args.seed >= 0:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

# model
mxfold2_predictor.model, _ = mxfold2_predictor.build_model(args)

# param and conf
if args.param != "":
    param = Path(args.param)
    if not param.exists() and conf is not None:
        param = Path(conf).parent / param
    p = torch.load(param, map_location="cpu")
    if isinstance(p, dict) and "model_state_dict" in p:
        p = p["model_state_dict"]
    mxfold2_predictor.model.load_state_dict(p)

# gpu
if args.gpu >= 0:
    mxfold2_predictor.model.to(torch.device("cuda", args.gpu))


def mxfold2_predict(seqs):
    tstart = time.time()

    if isinstance(seqs, str):
        seqs = [seqs]

    # clear memory
    if args.gpu >= 0:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a
        if r > 0.0 * t:
            torch.cuda.empty_cache()

    # predict
    scs = []
    preds = []
    bps = []
    mxfold2_predictor.test_loader = DataLoader(
        seqs, batch_size=1, shuffle=False
    )  # data loader
    mxfold2_predictor.model.eval()
    with torch.no_grad():
        for seq_batch in mxfold2_predictor.test_loader:
            scs_batch, preds_batch, bps_batch = mxfold2_predictor.model(seq_batch)
            scs += scs_batch.tolist()
            preds += preds_batch
            bps += bps_batch

    if args.gpu >= 0:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        memory = r / t
    else:
        r = t = memory = -1

    if len(preds) == 1:
        scs = scs[0]
        preds = preds[0]
        bps = bps[0]

    ttot = time.time() - tstart

    return preds, scs, bps, ttot, memory


def ufold_predict(seqs):
    tstart = time.time()

    if isinstance(seqs, str):
        seqs = [seqs]

    # clear memory
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a
    if r > 0.0 * t:
        torch.cuda.empty_cache()

    # prepare input file
    cwd = os.getcwd()
    os.chdir(path_ufold)
    input_path = Path("data/input.txt")
    output_path = Path("results/input_dot_ct_file.txt")
    with open(input_path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">{i}\n{s}\n")

    # predict
    main_ufold()
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    memory = r / t

    # read output
    with open(output_path, "r") as f:
        preds = f.read()
    preds = [s for s in preds.split("\n") if s]
    preds = preds[2::3]
    if len(preds) == 1:
        preds = preds[0]

    os.remove(input_path)
    os.remove(output_path)
    os.chdir(cwd)

    ttot = time.time() - tstart

    return preds, None, None, ttot, memory


def linearfold_predict(seqs):
    tstart = time.time()

    if isinstance(seqs, str):
        seqs = [seqs]

    # predict
    cwd = os.getcwd()
    os.chdir(path_linearfold)
    preds = [os.popen(f"echo {s} | ./linearfold").read() for s in seqs]

    # read output
    preds = [
        r.split("\n")[1].split()[0]
        if not r.startswith("Unrecognized")
        else "." * len(s)
        for s, r in zip(seqs, preds)
    ]
    if len(preds) == 1:
        preds = preds[0]

    os.chdir(cwd)

    ttot = time.time() - tstart

    return preds, None, None, ttot, 0.0


def rnapar_predict(seqs):
    tstart = time.time()

    if isinstance(seqs, str):
        seqs = [seqs]

    # predict
    cwd = os.getcwd()
    os.chdir(path_rnapar)
    os.popen(
        "python predict.py -i ./data/test.fasta -o ./predict/test.data -w ./models/weight-1.h5 -K 6 -C 61 -U 115 -N 53"
    )

    if len(preds) == 1:
        preds = preds[0]

    ttot = time.time() - tstart

    return preds, None, None, ttot, 0.0


def rnafold_predict(seq):
    tstart = time.time()
    output = os.popen(f"echo {seq} | RNAfold").read()
    pred = output.split("\n")[1].split(" ")[0]
    ttot = time.time() - tstart

    return pred, None, None, ttot, 0.0


def rnasubopt_predict(seq, kmax=5):
    tstart = time.time()
    output = os.popen(f"echo {seq} | RNAsubopt --sorted").read()
    lines = output.strip().split("\n")[1 : kmax + 1]
    pred = [(pred, float(energy)) for pred, energy in [l.split(" ") for l in lines]]
    ttot = time.time() - tstart

    return pred, None, None, ttot, 0.0


def probknot_predict(seq):
    tstart = time.time()
    suffix = datetime.datetime.now().strftime("%Y.%m.%d:%H.%M.%S:%f")
    path_in = f"temp_probknot_in_{suffix}.seq"
    path_middle = f"temp_probknot_middle_{suffix}.ct"
    path_out = f"temp_probknot_out_{suffix}.txt"
    seq = re.sub("[^ATCG]", "N", seq)
    with open(path_in, "w") as f:
        f.write(seq)

    os.popen(f"ProbKnot {path_in} {path_middle} --sequence").read()
    os.popen(f"ct2dot {path_middle} -1 {path_out}").read()
    pred = open(path_out, "r").read().split("\n")[2]

    os.remove(path_in)
    os.remove(path_middle)
    os.remove(path_out)
    ttot = time.time() - tstart

    return pred, None, None, ttot, 0.0


def oracle_get_cuts(struct):
    # Determine depth levels
    struct = re.sub("[^\(\)\.]", ".", struct)
    depths = []
    count = 0
    for c in struct:
        if c == "(":
            depths.append(count)
            count += 1
        elif c == ")":
            depths.append(count - 1)
            count -= 1
        else:
            depths.append(-1)
    depths = np.array(depths)

    # Determine sequence cuts
    cuts = []
    d = -1
    for d in range(max(depths) + 1):
        if np.count_nonzero(depths == d) == 2:
            continue

        bounds = np.where(depths == d)[0]
        if d > 0:
            outer_bounds = np.where(depths == d - 1)[0]
            bounds = np.array([outer_bounds[0]] + list(bounds) + [outer_bounds[1]])
        else:
            bounds = bounds[1:-1]
        cuts = [
            int(np.ceil((bounds[i] + bounds[i + 1]) / 2))
            for i in np.arange(len(bounds))[::2]
        ]

        break

    # Edge cases
    if not cuts:
        if max(depths) == -1:  # no pairs
            cuts = [int(len(struct) / 2)]
        else:  # only stacking concentric pairs
            gaps = np.array(
                [
                    len(depths)
                    - np.argmax(depths[::-1] == d)
                    - 1
                    - np.argmax(depths == d)
                    for d in range(max(depths) + 1)
                ]
            )
            too_small = gaps <= len(struct) / 2
            if np.any(too_small):
                d = np.argmax(too_small)
                bounds = np.where(depths == d)[0]
                outer_bounds = (
                    np.where(depths == d - 1)[0]
                    if d > 0
                    else np.array([0, len(struct)])
                )
                outer_gap = outer_bounds[1] - outer_bounds[0]
                lbda = (len(struct) / 2 - gaps[d]) / (outer_gap - gaps[d])
                cuts = [
                    int(np.ceil(x + lbda * (y - x)))
                    for x, y in zip(bounds, outer_bounds)
                ]
                cuts[1] = max(cuts[1], bounds[1] + 1)
            else:
                d = max(depths)
                bounds = np.where(depths == d)[0]
                margin = gaps[-1] - len(struct) / 2
                cuts = [
                    int(np.ceil(bounds[0] + margin / 2)),
                    int(np.ceil(bounds[1] - margin / 2)),
                ]
                d += 1  # we force entering an artificial additional depth level

    if cuts[0] == 0:
        cuts = cuts[1:]
    if cuts[-1] == len(struct):
        cuts = cuts[:-1]
    assert cuts

    outer = d > 0

    return cuts, outer


def divide_get_cuts(seq, min_height=0.28, min_distance=12, cut_model=default_model):
    seq_mat = format_data(seq).reshape((1, -1, 297))

    cuts = cut_model(seq_mat).numpy().ravel()
    min_height = min(min_height, max(cuts))

    def get_peaks(min_height):
        peaks = signal.find_peaks(cuts, height=min_height, distance=min_distance)[
            0
        ].tolist()
        if peaks and (peaks[0] == 0):
            peaks = peaks[1:]
        if peaks and (peaks[-1] == len(seq)):
            peaks = peaks[:-1]
        return peaks

    peaks = get_peaks(min_height)
    i = 0
    while len(peaks) < 2:
        if min_height < 0.01:
            peaks = []
            break
        min_height *= 0.9
        peaks = get_peaks(min_height)
    outer = True

    return peaks, outer


def linearfold_get_cuts(seq):
    preds, _, _, _, _ = linearfold_predict(seq)
    return oracle_get_cuts(preds)


def divide_predict(
    seq,
    max_length=1000,
    max_steps=None,
    cut_model=default_model,
    predict_fnc=mxfold2_predict,
    struct="",
    cuts_path=None,
    rna_name="",
):
    tstart = time.time()

    if len(seq) <= max_length or max_steps == 0:
        if cuts_path is not None:
            return "." * len(seq), None, None, 0.0, 0.0
        return predict_fnc(seq)

    if struct:
        cuts, outer = oracle_get_cuts(struct)
    else:
        cuts, outer = divide_get_cuts(seq, cut_model=cut_model)
    if cuts_path is not None:
        line = f'{rna_name.split("#Name: ")[1]},{seq},{str(cuts).replace(",", "")},{outer}\n'
        with open(cuts_path, "a") as f_out:
            f_out.write(line)

    # Cut sequence into subsequences
    random_cuts = [int(len(seq) / 3), int(len(seq) * 2 / 3)]
    if not cuts:
        cuts = random_cuts
    if cuts[0] > 0:
        cuts = [0] + cuts
    if cuts[-1] < len(seq):
        cuts = cuts + [len(seq)]
    if len(cuts) < (4 if outer else 3):
        cuts = [0] + random_cuts + [len(seq)]
    assert np.all(np.array(cuts)[1:] > np.array(cuts)[:-1])

    outer_bounds = []
    inner_bounds = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
    if outer:
        outer_bounds = [inner_bounds[0], inner_bounds[-1]]
        inner_bounds = inner_bounds[1:-1]

    # Predict subsequences
    preds = []
    outer_preds = []
    memories = []
    max_steps = max_steps - 1 if max_steps is not None else None
    for left_b, right_b in inner_bounds:
        subseq = seq[left_b:right_b]

        if struct:
            substruct = struct[left_b:right_b]
            assert substruct.count("(") == substruct.count(")")
            pred, _, _, _, memory = divide_predict(
                subseq,
                max_length=max_length,
                max_steps=max_steps,
                cut_model=cut_model,
                predict_fnc=predict_fnc,
                struct=substruct,
                cuts_path=cuts_path,
                rna_name=rna_name,
            )
        else:
            pred, _, _, _, memory = divide_predict(
                subseq,
                max_length=max_length,
                max_steps=max_steps,
                cut_model=cut_model,
                predict_fnc=predict_fnc,
                cuts_path=cuts_path,
                rna_name=rna_name,
            )

        preds.append(pred)
        memories.append(memory)

    if outer_bounds:
        left_subseq = seq[outer_bounds[0][0] : outer_bounds[0][1]]
        right_subseq = seq[outer_bounds[1][0] : outer_bounds[1][1]]
        subseq = left_subseq + right_subseq

        if struct:
            left_substruct = struct[outer_bounds[0][0] : outer_bounds[0][1]]
            right_substruct = struct[outer_bounds[1][0] : outer_bounds[1][1]]
            substruct = left_substruct + right_substruct
            assert substruct.count("(") == substruct.count(")")
            pred, _, _, _, memory = divide_predict(
                subseq,
                max_length=max_length,
                max_steps=max_steps,
                cut_model=cut_model,
                predict_fnc=predict_fnc,
                struct=substruct,
                cuts_path=cuts_path,
                rna_name=rna_name,
            )
        else:
            pred, _, _, _, memory = divide_predict(
                subseq,
                max_length=max_length,
                max_steps=max_steps,
                cut_model=cut_model,
                predict_fnc=predict_fnc,
                cuts_path=cuts_path,
                rna_name=rna_name,
            )

        left_pred, right_pred = pred[: len(left_subseq)], pred[len(left_subseq) :]
        outer_preds = [left_pred, right_pred]
        memories.append(memory)

    # Patch sub predictions into global prediction
    global_pred = "".join(preds)
    if outer_bounds:
        global_pred = outer_preds[0] + global_pred + outer_preds[1]
    memory = max(memories)
    ttot = time.time() - tstart

    return global_pred, None, None, ttot, memory
