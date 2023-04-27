import os
import sys
import random
from pathlib import Path
import time
import re
import numpy as np
from scipy import signal
import pickle

import torch
from torch.utils.data import DataLoader
from tensorflow import keras

from mxfold2.predict import Predict

sys.path.append('/mnt/e/Scripts')
sys.path.append('/mnt/e/Scripts/UFold')
from UFold.ufold_predict import main as main_ufold

from utils import format_data
from models import inv_exp_distance_to_cut_loss

my_model = keras.models.load_model(r'resources/models/model',
                                   custom_objects={'inv_exp_distance_to_cut_loss':
                                                   inv_exp_distance_to_cut_loss})


class Args:
    # self.func=<function Predict.add_args.<locals>.<lambda> at 0x7f526585d4c0>
    # self.input='bpRNA_CRW_1.fasta'
    seed = 0
    gpu = 0
    param = 'TrainSetAB.pth'
    result = None
    bpseq = None
    model = 'MixC'
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
    pair_join = 'cat'
    no_split_lr = False


mxfold2_predictor = Predict()
args = Args()
conf = '/mnt/e/Anaconda3Linux/lib/python3.9/site-packages/mxfold2/models/TrainSetAB.conf'

# seed
if args.seed >= 0:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

# model
mxfold2_predictor.model, _ = mxfold2_predictor.build_model(args)

# param and conf
if args.param != '':
    param = Path(args.param)
    if not param.exists() and conf is not None:
        param = Path(conf).parent / param
    p = torch.load(param, map_location='cpu')
    if isinstance(p, dict) and 'model_state_dict' in p:
        p = p['model_state_dict']
    mxfold2_predictor.model.load_state_dict(p)

# gpu
if args.gpu >= 0:
    mxfold2_predictor.model.to(torch.device("cuda", args.gpu))


def mxfold2_predict(seqs):
    if isinstance(seqs, str):
        seqs = [seqs]

    # clear memory
    if args.gpu >= 0:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a
        if r > 0. * t:
            torch.cuda.empty_cache()

    # data loader
    mxfold2_predictor.test_loader = DataLoader(seqs, batch_size=1, shuffle=False)

    # predict
    scs = []
    preds = []
    bps = []
    mxfold2_predictor.model.eval()
    tstart = time.time()
    with torch.no_grad():
        for seq_batch in mxfold2_predictor.test_loader:
            scs_batch, preds_batch, bps_batch = mxfold2_predictor.model(seq_batch)
            scs += scs_batch.tolist()
            preds += preds_batch
            bps += bps_batch
    ttot = time.time() - tstart
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    memory = r / t

    if len(preds) == 1:
        scs = scs[0]
        preds = preds[0]
        bps = bps[0]

    return preds, scs, bps, ttot, memory


def ufold_predict(seqs):
    if isinstance(seqs, str):
        seqs = [seqs]

    # clear memory
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a
    if r > 0. * t:
        torch.cuda.empty_cache()

    # prepare input file
    cwd = os.getcwd()
    os.chdir(r'/mnt/e/Scripts/UFold')
    input_path = os.path.join('data', 'input.txt')
    output_path = os.path.join('results', 'input_dot_ct_file.txt')
    with open(input_path, 'w') as f:
        for i, s in enumerate(seqs):
            f.write(f'>{i}\n{s}\n')

    # predict
    tstart = time.time()
    main_ufold()
    ttot = time.time() - tstart
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    memory = r / t

    # read output
    with open(output_path, 'r') as f:
        preds = f.read()
    preds = [s for s in preds.split('\n') if s]
    preds = preds[2::3]
    if len(preds) == 1:
        preds = preds[0]

    os.remove(input_path)
    os.remove(output_path)
    os.chdir(cwd)

    return preds, None, None, ttot, memory


def linearfold_predict(seqs):
    if isinstance(seqs, str):
        seqs = [seqs]

    # predict
    cwd = os.getcwd()
    os.chdir(r'/mnt/e/Scripts/LinearFold')
    tstart = time.time()
    preds = [os.popen(f'echo {s} | ./linearfold').read() for s in seqs]
    ttot = time.time() - tstart

    # read output
    preds = [r.split('\n')[1].split()[0] for r in preds]
    if len(preds) == 1:
        preds = preds[0]

    os.chdir(cwd)

    return preds, None, None, ttot, 0.


def divide_predict(seq, max_length=200, struct='', store_cuts=False, cuts_file=None,
                                                                     rna_name=''):
    if len(seq) <= max_length:
        if cuts_file is not None:
            return '.' * len(seq), None, None, 0., 0.
        return mxfold2_predict(seq)

    if struct:
        cuts, outer = divide_get_cuts_cheat(struct)
    else:
        cuts, outer = divide_get_cuts(seq)
    if cuts_file is not None:
        line = f'{rna_name.split("#Name: ")[1]},{seq},{str(cuts).replace(",", "")},{outer}\n'
        cuts_file.write(line)

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
    inner_bounds = [(cuts[i], cuts[i+1]) for i in range(len(cuts) - 1)]
    if outer:
        outer_bounds = [inner_bounds[0], inner_bounds[-1]]
        inner_bounds = inner_bounds[1:-1]

    # Predict subsequences
    preds = []
    outer_preds = []
    times = []
    memories = []
    for left_b, right_b in inner_bounds:
        subseq = seq[left_b:right_b]

        if struct:
            substruct = struct[left_b:right_b]
            assert substruct.count('(') == substruct.count(')')
            pred, _, _, ttot, memory = divide_predict(subseq, max_length=max_length,
                                                              struct=substruct,
                                                              cuts_file=cuts_file,
                                                              rna_name=rna_name)
        else:
            pred, _, _, ttot, memory = divide_predict(subseq, max_length=max_length,
                                                              cuts_file=cuts_file,
                                                              rna_name=rna_name)

        preds.append(pred)
        times.append(ttot)
        memories.append(memory)

    if outer_bounds:
        left_subseq = seq[outer_bounds[0][0]:outer_bounds[0][1]]
        right_subseq = seq[outer_bounds[1][0]:outer_bounds[1][1]]
        subseq = left_subseq + right_subseq

        if struct:
            left_substruct = struct[outer_bounds[0][0]:outer_bounds[0][1]]
            right_substruct = struct[outer_bounds[1][0]:outer_bounds[1][1]]
            substruct = left_substruct + right_substruct
            assert substruct.count('(') == substruct.count(')')
            pred, _, _, ttot, memory = divide_predict(subseq, max_length=max_length,
                                                              struct=substruct,
                                                              cuts_file=cuts_file,
                                                              rna_name=rna_name)
        else:
            pred, _, _, ttot, memory = divide_predict(subseq, max_length=max_length,
                                                              cuts_file=cuts_file,
                                                              rna_name=rna_name)

        left_pred, right_pred = pred[:len(left_subseq)], pred[len(left_subseq):]
        outer_preds = [left_pred, right_pred]
        times.append(ttot)
        memories.append(memory)

    # Patch sub predictions into global prediction
    global_pred = ''.join(preds)
    if outer_bounds:
        global_pred = outer_preds[0] + global_pred + outer_preds[1]
    ttot = sum(times)
    memory = max(memories) # could also try sum of memories

    return global_pred, None, None, ttot, memory


def divide_get_cuts_cheat(struct):
    # Determine depth levels
    struct = re.sub('[^\(\)\.]', '.', struct)
    depths = []
    count = 0
    for c in struct:
        if c == '(':
            depths.append(count)
            count += 1
        elif c == ')':
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
            outer_bounds = np.where(depths == d-1)[0]
            bounds = np.array([outer_bounds[0]] + list(bounds) + [outer_bounds[1]])
        else:
            bounds = bounds[1:-1]
        cuts = [int(np.ceil((bounds[i] + bounds[i+1]) / 2))
                        for i in np.arange(len(bounds))[::2]]

        break

    # Edge cases
    if not cuts:
        if max(depths) == -1: # no pairs
            cuts = [int(len(struct) / 2)]
        else: # only stacking concentric pairs
            gaps = np.array([len(depths) - np.argmax(depths[::-1] == d) - 1
                             - np.argmax(depths == d)
                             for d in range(max(depths) + 1)])
            too_small = gaps <= len(struct) / 2
            if np.any(too_small):
                d = np.argmax(too_small)
                bounds = np.where(depths == d)[0]
                outer_bounds = np.where(depths == d-1)[0] if d > 0 \
                                    else np.array([0, len(struct)])
                outer_gap = outer_bounds[1] - outer_bounds[0]
                lbda = (len(struct) / 2 - gaps[d]) / (outer_gap - gaps[d])
                cuts = [int(np.ceil(x + lbda * (y - x)))
                        for x, y in zip(bounds, outer_bounds)]
                cuts[1] = max(cuts[1], bounds[1] + 1)
            else:
                d = max(depths)
                bounds = np.where(depths == d)[0]
                margin = gaps[-1] - len(struct) / 2
                cuts = [int(np.ceil(bounds[0] + margin / 2)),
                        int(np.ceil(bounds[1] - margin / 2))]
                d += 1 # we force entering an artificial additional depth level

    if cuts[0] == 0:
        cuts = cuts[1:]
    if cuts[-1] == len(struct):
        cuts = cuts[:-1]
    assert cuts

    outer = d > 0

    return cuts, outer


def divide_get_cuts(seq, min_height=0.33, min_distance=15):
    seq_mat = format_data(seq).reshape((1, -1, 297))

    cuts = my_model(seq_mat).numpy().ravel()
    min_height = min(min_height, max(cuts))

    def get_peaks(min_height):
        peaks = signal.find_peaks(cuts, height=min_height, distance=min_distance)[0].tolist()
        if not peaks:
            return peaks
        if peaks[0] == 0:
            peaks = peaks[1:]
        if peaks[-1] == len(seq):
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
