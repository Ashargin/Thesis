import os
import subprocess
import random
from pathlib import Path
import time
import datetime
import re
import numpy as np
import scipy.signal
import itertools
import signal

import torch
import keras

from src.utils import (
    format_data,
    eval_energy,
    get_structure_scores,
    pairs_to_struct,
    optimize_pseudoknots,
)

# Settings
DEFAULT_CUT_MODEL = (
    Path(__file__).parents[1] / "resources/models/CNN1D_1600EVOAUGINCRANGE.keras"
)

# Load cut model
default_cut_model = keras.models.load_model(DEFAULT_CUT_MODEL)

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("The program was terminated because it exceeded timeout.")


signal.signal(signal.SIGALRM, timeout_handler)


# Prediction functions
def mxfold2_predict(
    seq, path_mxfold2="../mxfold2", conf="TR0-canonicals.conf", timeout=None
):
    # path_mxfold2 is the path to the mxfold2 repository
    # https://github.com/mxfold/mxfold2

    tstart = time.time()
    if timeout is not None:
        signal.alarm(timeout)
    path_mxfold2 = Path(path_mxfold2)

    # clear memory
    torch.cuda.empty_cache()

    suffix = datetime.datetime.now().strftime("%Y.%m.%d:%H.%M.%S:%f")
    path_in = f"temp_mxfold2_in_{suffix}.fa"
    with open(path_in, "w") as f:
        f.write(f">0\n{seq}\n")

    res = os.popen(
        f"mxfold2 predict @{path_mxfold2 / 'models' / conf} {path_in}"
    ).read()
    pred = res.split("\n")[2].split(" ")[0]
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    memory = r / t

    os.remove(path_in)

    if timeout is not None:
        signal.alarm(0)
    ttot = time.time() - tstart

    return pred, ttot, memory


def ufold_predict(seq, path_ufold="../UFold", timeout=None):
    # path_ufold is the path to the UFold repository
    # https://github.com/uci-cbcl/UFold

    # UFold rejects sequences longer then 600 nt
    if len(seq) > 600:
        raise ValueError(
            f"Discarded sequence of length {len(seq)}. UFold only accepts sequences shorter than 600 nt."
        )

    # UFold only accept A, U, C, G bases
    authorized_bases = ["A", "U", "C", "G"]
    seq_bases = list(seq)
    for i, b in enumerate(seq_bases):
        if b not in authorized_bases:
            seq_bases[i] = np.random.choice(authorized_bases)
    seq = "".join(seq_bases)

    tstart = time.time()
    if timeout is not None:
        signal.alarm(timeout)
    path_ufold = Path(path_ufold)

    # clear memory
    torch.cuda.empty_cache()

    # prepare input file
    suffix = datetime.datetime.now().strftime("%Y.%m.%d:%H.%M.%S:%f")
    temp_rna_name = f"TEMP_RNA_NAME_{suffix}"
    path_in = path_ufold / "data" / "input.txt"
    path_out = path_ufold / "results" / "save_ct_file" / f"{temp_rna_name}.ct"
    with open(path_in, "w") as f:
        f.write(f">{temp_rna_name}\n{seq}\n")

    # predict
    subprocess.run(["python", "ufold_predict.py"], cwd=path_ufold)
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    memory = r / t

    # read output
    with open(path_out, "r") as f:
        pred_txt = f.read()
    pred_lines = [s for s in pred_txt.split("\n") if s][1:]
    pairs = np.array([int(l.split("\t")[4]) for l in pred_lines])

    # eliminate occasional conflicts from UFold output
    pairs = np.array([j if pairs[j - 1] - 1 == i else 0 for i, j in enumerate(pairs)])
    pred = pairs_to_struct(pairs)

    os.remove(path_in)
    os.remove(path_out)

    if timeout is not None:
        signal.alarm(0)
    ttot = time.time() - tstart

    return pred, ttot, memory


def linearfold_predict(seq, path_linearfold="../LinearFold", timeout=None):
    # path_linearfold is the path to the LinearFold repository
    # https://github.com/LinearFold/LinearFold

    tstart = time.time()
    if timeout is not None:
        signal.alarm(timeout)
    path_linearfold = Path(path_linearfold)

    # predict
    pred = os.popen(f"echo {seq} | {path_linearfold / 'linearfold'}").read()

    # read output
    pred = (
        pred.split("\n")[1].split()[0]
        if not pred.startswith("Unrecognized")
        else "." * len(seq)
    )

    if timeout is not None:
        signal.alarm(0)
    ttot = time.time() - tstart

    return pred, ttot, 0.0


def rnafold_predict(seq, timeout=None):
    # https://www.tbi.univie.ac.at/RNA/

    tstart = time.time()
    if timeout is not None:
        signal.alarm(timeout)

    output = os.popen(f"echo {seq} | RNAfold").read()
    pred = output.split("\n")[1].split(" ")[0]

    if timeout is not None:
        signal.alarm(0)
    ttot = time.time() - tstart

    return pred, ttot, 0.0


def rnasubopt_predict(seq, kmax=5, delta=0.1, timeout=None):
    # https://www.tbi.univie.ac.at/RNA/

    tstart = time.time()
    if timeout is not None:
        signal.alarm(timeout)

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

    if timeout is not None:
        signal.alarm(0)
    ttot = time.time() - tstart

    return preds, ttot, 0.0


def probknot_predict(seq, path_rnastructure="../RNAstructure", timeout="meta"):
    # path_rnastructure is the path to the RNAstructure folder
    # ProbKnot is part of the RNAstructure package
    # https://rna.urmc.rochester.edu/RNAstructure.html

    tstart = time.time()
    if timeout is not None:
        if timeout == "meta":
            timeout = 50 + (len(seq) > 1000) * 150 + (len(seq) > 2000) * 400
        signal.alarm(timeout)
    path_rnastructure = Path(path_rnastructure)

    suffix = datetime.datetime.now().strftime("%Y.%m.%d:%H.%M.%S:%f")
    path_in = f"temp_probknot_in_{suffix}.seq"
    path_middle = f"temp_probknot_middle_{suffix}.ct"
    path_out = f"temp_probknot_out_{suffix}.txt"
    seq = re.sub("[^AUCG]", "N", seq)
    with open(path_in, "w") as f:
        f.write(seq)

    try:
        os.popen(
            f"{path_rnastructure / 'exe' / 'ProbKnot'} {path_in} {path_middle} --sequence"
        ).read()
    except TimeoutError:
        os.remove(path_in)
        timeout_handler(None, None)
    try:
        os.popen(
            f"{path_rnastructure / 'exe' / 'ct2dot'} {path_middle} -1 {path_out}"
        ).read()
    except TimeoutError:
        os.remove(path_in)
        os.remove(path_middle)
        timeout_handler(None, None)
    pred = open(path_out, "r").read().split("\n")[2]

    os.remove(path_in)
    os.remove(path_middle)
    os.remove(path_out)
    if timeout is not None:
        signal.alarm(0)
    ttot = time.time() - tstart

    return pred, ttot, 0.0


def ensemble_predict(seq, predict_fncs, timeout=None):
    tstart = time.time()
    if timeout is not None:
        signal.alarm(timeout)

    preds = []
    memories = []
    for pred_f in predict_fncs:
        pred, _, mem = pred_f(seq)
        preds.append(pred)
        memories.append(mem)

    energies = [eval_energy(seq, pred) for pred in preds]

    pred_energies = list(zip(preds, energies))
    pred_energies.sort(key=lambda x: x[1])

    memory = max(memories)

    if timeout is not None:
        signal.alarm(0)
    ttot = time.time() - tstart

    return pred_energies, ttot, memory


def knotfold_predict(seq, path_knotfold="../KnotFold", timeout=None):
    # path_knotfold is the path to the KnotFold repository
    # https://github.com/gongtiansu/KnotFold

    if len(seq) == 1:  # to avoid IndexError from KnotFold for sequences of length 1
        return ".", 0.0, 0.0

    tstart = time.time()
    if timeout is not None:
        signal.alarm(timeout)
    path_knotfold = Path(path_knotfold)

    # clear memory
    torch.cuda.empty_cache()

    # prepare input file
    suffix = datetime.datetime.now().strftime("%Y.%m.%d:%H.%M.%S:%f")
    temp_rna_name = f"TEMP_RNA_NAME_{suffix}"
    path_in = f"temp_knotfold_in_{suffix}.fasta"
    path_out = f"{temp_rna_name}.bpseq"
    with open(path_knotfold / path_in, "w") as f:
        f.write(f">{temp_rna_name}\n{seq}\n")

    # predict
    res = subprocess.run(
        ["python", "KnotFold.py", "-i", path_in, "-o", ".", "--cuda"], cwd=path_knotfold
    )
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    memory = r / t

    # read output
    os.remove(path_knotfold / path_in)
    if res.returncode != 0:
        raise MemoryError(
            f"The KnotFold script could not run properly. The input sequence may be too long. Avoid sequences longer than 2000 nucleotides when using KnotFold. If the sequence is shorter, then something else in the KnotFold script may have caused this error. Look in the traceback from the KnotFold script for more information."
        )
    with open(path_knotfold / path_out, "r") as f:
        pred_txt = f.read()
    pairs = np.array(
        [int(line.split(" ")[-1]) for line in pred_txt.strip().split("\n")]
    )

    # eliminate occasional conflicts from KnotFold output
    pairs = np.array([j if pairs[j - 1] - 1 == i else 0 for i, j in enumerate(pairs)])
    pred = pairs_to_struct(pairs)

    os.remove(path_knotfold / path_out)

    if timeout is not None:
        signal.alarm(0)
    ttot = time.time() - tstart

    return pred, ttot, memory


def pkiss_predict(seq, timeout="meta"):
    # https://bibiserv.cebitec.uni-bielefeld.de/pkiss

    tstart = time.time()
    if timeout is not None:
        if timeout == "meta":
            timeout = 50 + (len(seq) > 1000) * 150 + (len(seq) > 2000) * 400
        signal.alarm(timeout)

    # pKiss only accepts A, U, G, C nucleotides
    pred = ["" if c in ["A", "U", "C", "G"] else "." for c in seq]
    seq = re.sub("[^AUCG]", "", seq)

    res = os.popen(f"pKiss --mode=mfe {seq}").read()
    sub_pred = res.split("\n")[2].split(" ")[-1]

    i = 0
    for p in sub_pred:
        while pred[i] == ".":
            i += 1
        pred[i] = p
        i += 1
    pred = "".join(pred)

    if timeout is not None:
        signal.alarm(0)
    ttot = time.time() - tstart

    return pred, ttot, 0.0


def ipknot_predict(seq, path_ipknot="../ipknot", timeout=None):
    # https://github.com/satoken/ipknot

    tstart = time.time()
    if timeout is not None:
        if timeout == "meta":
            timeout = 50 + (len(seq) > 1000) * 150 + (len(seq) > 2000) * 400
        signal.alarm(timeout)
    path_ipknot = Path(path_ipknot)

    suffix = datetime.datetime.now().strftime("%Y.%m.%d:%H.%M.%S:%f")
    path_in = f"temp_ipknot_in_{suffix}.fa"
    with open(path_in, "w") as f:
        f.write(f">0\n{seq}\n")

    try:
        res = os.popen(f"{path_ipknot / 'bin' / 'ipknot'} {path_in}").read()
    except TimeoutError:
        os.remove(path_in)
        timeout_handler(None, None)
    pred = res.split("\n")[2]

    os.remove(path_in)

    if timeout is not None:
        signal.alarm(0)
    ttot = time.time() - tstart

    return pred, ttot, 0.0


def oracle_get_cuts(struct):
    if len(struct) <= 3:
        return [], True

    # Determine depth levels
    struct = re.sub(r"[^\(\)\.]", ".", struct)
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


def dividefold_get_cuts(
    seq,
    min_height=0.28,
    min_distance=12,
    cut_model=default_cut_model,
    max_motifs=200,
    fuse_to=None,
):
    seq_mat = format_data(seq, max_motifs=max_motifs)[np.newaxis, :, :]

    cuts = cut_model(seq_mat).detach().cpu().numpy().ravel()
    min_height = min(min_height, max(cuts))

    def get_peaks(min_height):
        peaks = scipy.signal.find_peaks(cuts, height=min_height, distance=min_distance)[
            0
        ]
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
    preds, _, _ = linearfold_predict(seq)
    return oracle_get_cuts(preds)


def dividefold_get_fragment_ranges_preds(
    seq,
    max_length=1000,
    max_steps=None,
    min_steps=0,
    cut_model=default_cut_model,
    predict_fnc=linearfold_predict,
    max_motifs=200,
    fuse_to=None,
    struct="",
    return_cuts=False,
):
    tstart = time.time()

    if max_steps == 0 or len(seq) <= max_length and min_steps <= 0:
        pred, ttot, memory = (
            predict_fnc(seq) if not return_cuts else ("." * len(seq), 0.0, 0.0)
        )
        frag_preds = [(np.array([[0, len(seq) - 1]]).astype(int), pred)]
        return frag_preds, ttot, memory

    if struct:
        cuts, outer = oracle_get_cuts(struct)
    else:
        cuts, outer = dividefold_get_cuts(
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

        substruct = ""
        if struct:
            substruct = struct[left_b:right_b]
            assert substruct.count("(") == substruct.count(")")
        this_frag_preds, _, memory = dividefold_get_fragment_ranges_preds(
            subseq,
            max_length=max_length,
            max_steps=max_steps,
            min_steps=min_steps,
            cut_model=cut_model,
            predict_fnc=predict_fnc,
            max_motifs=max_motifs,
            fuse_to=fuse_to,
            struct=substruct,
            return_cuts=return_cuts,
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

        substruct = ""
        if struct:
            left_substruct = struct[left_b_1:right_b_1]
            right_substruct = struct[left_b_2:right_b_2]
            substruct = left_substruct + right_substruct
            assert substruct.count("(") == substruct.count(")")
        this_frag_preds, _, memory = dividefold_get_fragment_ranges_preds(
            subseq,
            max_length=max_length,
            max_steps=max_steps,
            min_steps=min_steps,
            cut_model=cut_model,
            predict_fnc=predict_fnc,
            max_motifs=max_motifs,
            fuse_to=fuse_to,
            struct=substruct,
            return_cuts=return_cuts,
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

    return frag_preds, ttot, memory


def dividefold_predict(
    seq,
    max_length=None,
    max_steps=None,
    min_steps=0,
    multipred_kmax=20,
    cut_model=default_cut_model,
    predict_fnc=linearfold_predict,
    max_motifs=200,
    fuse_to=None,
    struct="",
    struct_to_print_fscores="",
    return_cuts=False,
):
    tstart = time.time()

    if isinstance(predict_fnc, list):
        predict_fnc = lambda seq: ensemble_predict(seq, predict_fncs=predict_fnc)

    if max_length is None:
        if (predict_fnc is None) or (predict_fnc.__name__ != "knotfold_predict"):
            max_length = 2000 if len(seq) > 2500 else 400
        else:
            max_length = 1000

    if max_steps is not None and max_steps < min_steps:
        raise ValueError("max_steps must be greater than min_steps.")

    if struct:
        struct = optimize_pseudoknots(struct)
    if struct_to_print_fscores:
        struct_to_print_fscores = optimize_pseudoknots(struct_to_print_fscores)

    frag_preds, _, memory = dividefold_get_fragment_ranges_preds(
        seq,
        max_length=max_length,
        max_steps=max_steps,
        min_steps=min_steps,
        cut_model=cut_model,
        predict_fnc=predict_fnc,
        max_motifs=max_motifs,
        fuse_to=fuse_to,
        struct=struct,
        return_cuts=return_cuts,
    )

    if return_cuts:
        ttot = time.time() - tstart
        return frag_preds, ttot, memory

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
                _, _, fscore, _ = get_structure_scores(struct_to_print_fscores, p)
                print((fscore, e))

    else:  # single prediction function
        global_pred = assemble_fragments(frag_preds)

    ttot = time.time() - tstart
    return global_pred, ttot, memory
