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
import itertools

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

from src.utils import format_data, eval_energy, get_scores
from src.models.loss import inv_exp_distance_to_cut_loss

default_cut_model = keras.models.load_model(
    Path("resources/models/CNN1D_sequencewise_200motifs256dilINV_augmented"),
    compile=False,
)
default_cut_model.compile(
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


def rnasubopt_predict(seq, kmax=5, delta=0.1):
    tstart = time.time()
    output = os.popen(f"echo {seq} | RNAsubopt --sorted").read()
    lines = output.strip().split("\n")[1:]
    all_preds = [
        (pr, float(e)) for pr, e in [[x for x in l.split(" ") if x] for l in lines]
    ]

    # Filter results
    energy = -np.inf
    selected = 0
    preds = []
    for pr, e in all_preds:
        if e >= energy + delta:
            preds.append((pr, e))
            energy = e
            selected += 1
            if selected >= kmax:
                break

    ttot = time.time() - tstart

    return preds, None, None, ttot, 0.0


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


def ensemble_predict(seq):
    tstart = time.time()

    pred_mx, _, _, _, mem_mx = mxfold2_predict(seq)
    pred_lf, _, _, _, mem_lf = linearfold_predict(seq)
    pred_rnaf, _, _, _, mem_rnaf = rnafold_predict(seq)

    energy_mx = eval_energy(seq, pred_mx)
    energy_lf = eval_energy(seq, pred_lf)
    energy_rnaf = eval_energy(seq, pred_rnaf)

    preds = [(pred_mx, energy_mx), (pred_lf, energy_lf), (pred_rnaf, energy_rnaf)]
    preds.sort(key=lambda x: x[1])

    memory = max([mem_mx, mem_lf, mem_rnaf])
    ttot = time.time() - tstart

    return preds, None, None, ttot, memory


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


def divide_get_cuts(
    seq,
    min_height=0.28,
    min_distance=12,
    cut_model=default_cut_model,
    max_motifs=None,
    fuse_to=None,
):
    seq_mat = format_data(seq, max_motifs=max_motifs)[np.newaxis, :, :]

    cuts = cut_model(seq_mat).numpy().ravel()
    min_height = min(min_height, max(cuts))

    def get_peaks(min_height):
        peaks = signal.find_peaks(cuts, height=min_height, distance=min_distance)[0]
        if peaks.size > 0 and (peaks[0] == 0):
            peaks = peaks[1:]
        if peaks.size > 0 and (peaks[-1] == len(seq)):
            peaks = peaks[:-1]
        return peaks

    peaks = get_peaks(min_height)
    while len(peaks) < 2:
        if min_height < 0.01:
            peaks = np.zeros((0,))
            break
        min_height *= 0.9
        peaks = get_peaks(min_height)
    outer = True

    def fuse_consecutive_peaks(peak_array):
        if len(peak_array) <= 2:
            return peak_array

        for n_inner_frags in range(1, len(peak_array)):
            bounds = []
            losses = []
            for inner_cuts in itertools.combinations(
                peak_array[1:-1], n_inner_frags - 1
            ):
                this_bounds = np.concatenate(
                    [
                        [peak_array[0]],
                        inner_cuts,
                        [peak_array[-1]],
                    ]
                ).astype(int)
                if not np.all(this_bounds[1:] - this_bounds[:-1] <= fuse_to):
                    continue

                this_loss = np.sum(
                    (
                        (this_bounds[1:] - this_bounds[:-1])
                        / (peak_array[-1] - peak_array[0])
                    )
                    ** 2
                )
                bounds.append(this_bounds)
                losses.append(this_loss)

            if bounds:
                best_bounds = bounds[np.argmin(losses)]
                return best_bounds

    def fuse_peaks(peak_array):
        large_gaps_idx = np.concatenate(
            [
                [0],
                np.argwhere(peak_array[1:] - peak_array[:-1] > fuse_to).ravel() + 1,
                [len(peak_array)],
            ]
        )
        fusables = [
            peak_array[start:end]
            for start, end in zip(large_gaps_idx[:-1], large_gaps_idx[1:])
        ]
        fused = [fuse_consecutive_peaks(peak_subarray) for peak_subarray in fusables]
        return np.concatenate(fused)

    if peaks.size > 0 and fuse_to is not None:
        peaks = fuse_peaks(peaks)

    return peaks.tolist(), outer


def linearfold_get_cuts(seq):
    preds, _, _, _, _ = linearfold_predict(seq)
    return oracle_get_cuts(preds)


def divide_get_fragment_ranges_preds(
    seq,
    max_length=1000,
    max_steps=None,
    min_steps=0,
    cut_model=default_cut_model,
    predict_fnc=mxfold2_predict,
    max_motifs=None,
    fuse_to=None,
    struct="",
    evaluate_cutting_model=False,
):
    tstart = time.time()

    if max_steps == 0 or len(seq) <= max_length and min_steps <= 0:
        pred, a, b, ttot, memory = (
            predict_fnc(seq)
            if not evaluate_cutting_model
            else ("." * len(seq), None, None, 0.0, 0.0)
        )
        frag_preds = [(np.array([[0, len(seq) - 1]]).astype(int), pred)]
        return frag_preds, a, b, ttot, memory

    if struct:
        cuts, outer = oracle_get_cuts(struct)
    else:
        cuts, outer = divide_get_cuts(
            seq, cut_model=cut_model, max_motifs=max_motifs, fuse_to=fuse_to
        )

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
    frag_preds = []
    memories = []
    max_steps = max_steps - 1 if max_steps is not None else None
    min_steps -= 1
    for left_b, right_b in inner_bounds:
        subseq = seq[left_b:right_b]

        if struct:
            substruct = struct[left_b:right_b]
            assert substruct.count("(") == substruct.count(")")
            this_frag_preds, _, _, _, memory = divide_get_fragment_ranges_preds(
                subseq,
                max_length=max_length,
                max_steps=max_steps,
                min_steps=min_steps,
                cut_model=cut_model,
                predict_fnc=predict_fnc,
                max_motifs=max_motifs,
                fuse_to=fuse_to,
                struct=substruct,
                evaluate_cutting_model=evaluate_cutting_model,
            )
        else:
            this_frag_preds, _, _, _, memory = divide_get_fragment_ranges_preds(
                subseq,
                max_length=max_length,
                max_steps=max_steps,
                min_steps=min_steps,
                cut_model=cut_model,
                predict_fnc=predict_fnc,
                max_motifs=max_motifs,
                fuse_to=fuse_to,
                evaluate_cutting_model=evaluate_cutting_model,
            )

        for _range, pred in this_frag_preds:
            frag_preds.append((_range + left_b, pred))
        memories.append(memory)

    if outer_bounds:
        left_b_1, right_b_1 = outer_bounds[0]
        left_b_2, right_b_2 = outer_bounds[1]
        left_subseq = seq[left_b_1:right_b_1]
        right_subseq = seq[left_b_2:right_b_2]
        subseq = left_subseq + right_subseq

        if struct:
            left_substruct = struct[left_b_1:right_b_1]
            right_substruct = struct[left_b_2:right_b_2]
            substruct = left_substruct + right_substruct
            assert substruct.count("(") == substruct.count(")")
            this_frag_preds, _, _, _, memory = divide_get_fragment_ranges_preds(
                subseq,
                max_length=max_length,
                max_steps=max_steps,
                min_steps=min_steps,
                cut_model=cut_model,
                predict_fnc=predict_fnc,
                max_motifs=max_motifs,
                fuse_to=fuse_to,
                struct=substruct,
                evaluate_cutting_model=evaluate_cutting_model,
            )
        else:
            this_frag_preds, _, _, _, memory = divide_get_fragment_ranges_preds(
                subseq,
                max_length=max_length,
                max_steps=max_steps,
                min_steps=min_steps,
                cut_model=cut_model,
                predict_fnc=predict_fnc,
                max_motifs=max_motifs,
                fuse_to=fuse_to,
                evaluate_cutting_model=evaluate_cutting_model,
            )

        sep = right_b_1 - left_b_1
        for _range, pred in this_frag_preds:
            lefts = _range[_range[:, 1] < sep]
            middle = _range[np.all([_range[:, 0] < sep, _range[:, 1] >= sep], axis=0)]
            rights = _range[_range[:, 0] >= sep]
            middle_left = (
                np.array([[middle[0, 0], sep - 1]])
                if middle.size > 0
                else np.zeros((0, 2))
            )
            middle_right = (
                np.array([[sep, middle[0, 1]]]) if middle.size > 0 else np.zeros((0, 2))
            )
            new_range = np.vstack(
                [
                    lefts + left_b_1,
                    middle_left + left_b_1,
                    middle_right + left_b_2 - sep,
                    rights + left_b_2 - sep,
                ]
            )
            frag_preds.append((new_range.astype(int), pred))
        memories.append(memory)

    # Patch sub predictions into global prediction
    memory = max(memories)
    ttot = time.time() - tstart

    return frag_preds, None, None, ttot, memory


def divide_predict(
    seq,
    max_length=None,
    max_steps=None,
    min_steps=None,
    multipred_kmax=20,
    cut_model=default_cut_model,
    predict_fnc=mxfold2_predict,
    max_motifs=200,
    fuse_to=None,
    struct="",
    struct_to_print_fscores="",
    evaluate_cutting_model=False,
):
    tstart = time.time()

    if min_steps is None:
        min_steps = 1 if len(seq) >= 400 else 0
        if max_length is not None:
            min_steps = 0
        if max_steps is not None:
            min_steps = min(min_steps, max_steps)
    if max_length is None:
        max_length = 2000 if len(seq) < 1300 or len(seq) >= 1600 else 200

    if max_steps is not None and max_steps < min_steps:
        raise Warning("max_steps must be greater than min_steps.")

    frag_preds, _, _, _, memory = divide_get_fragment_ranges_preds(
        seq,
        max_length=max_length,
        max_steps=max_steps,
        min_steps=min_steps,
        cut_model=cut_model,
        predict_fnc=predict_fnc,
        max_motifs=max_motifs,
        fuse_to=fuse_to,
        struct=struct,
        evaluate_cutting_model=evaluate_cutting_model,
    )

    if evaluate_cutting_model:
        return frag_preds, None, None, None, None

    def assemble_fragments(in_frag_preds):
        connex_frags = []
        for _range, pred in in_frag_preds:
            fragment_pred = pred
            for start, end in _range:
                part_pred = fragment_pred[: end - start + 1]
                fragment_pred = fragment_pred[end - start + 1 :]
                connex_frags.append((start, end, part_pred))
        connex_frags.sort(key=lambda x: x[0])
        out_global_pred = "".join([pred for start, range, pred in connex_frags])
        return out_global_pred

    def find(tsum, mpreds):
        if len(mpreds) == 1:
            for pred, val in mpreds[0]:
                if val == tsum:
                    yield [pred]
            return
        for pred, val in mpreds[0]:
            if val <= tsum:
                for f in find(tsum - val, mpreds[1:]):
                    yield [pred] + f
        return

    if isinstance(frag_preds[0][1], list):  # multiple predictions function
        ranges, multipreds = zip(*frag_preds)
        multipreds = [
            [(pred, round(10 * energy)) for pred, energy in multi]
            for multi in multipreds
        ]
        energy_mins = [min([energy for pred, energy in multi]) for multi in multipreds]
        multipreds = [
            [(pred, energy - energy_mins[i]) for pred, energy in multi]
            for i, multi in enumerate(multipreds)
        ]
        target_energy = 0
        selected_frag_preds = []
        n_selected = 0
        max_energy = sum(
            [max([energy for pred, energy in multi]) for multi in multipreds]
        )
        while True:
            for match in find(target_energy, multipreds):
                selected_frag_preds.append(match)
                n_selected += 1
                if n_selected >= multipred_kmax:
                    break
            else:
                target_energy += 1
                if target_energy > max_energy:
                    break
                continue
            break
        all_frag_preds = [
            list(zip(ranges, this_selected)) for this_selected in selected_frag_preds
        ]
        all_global_preds = [
            assemble_fragments(this_frag_preds) for this_frag_preds in all_frag_preds
        ]
        global_energies = [eval_energy(seq, pred) for pred in all_global_preds]
        pred_energies = list(
            sorted(zip(all_global_preds, global_energies), key=lambda x: x[1])
        )
        all_global_preds, global_energies = zip(*pred_energies)
        global_pred = all_global_preds[0]

        if struct_to_print_fscores:
            for p, e in zip(all_global_preds, global_energies):
                _, _, fscore, _ = get_scores(struct_to_print_fscores, p)
                print((fscore, e))

    else:  # single prediction function
        global_pred = assemble_fragments(frag_preds)

    ttot = time.time() - tstart
    return global_pred, None, None, ttot, memory
