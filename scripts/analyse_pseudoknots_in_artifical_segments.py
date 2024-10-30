import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from src.utils import struct_to_pairs

df = pd.read_csv(
    r"C:\Work\Thesis\resources\results\cutting_metrics\sequencewise\dividefold_cnn_1000_sequencewise_cuttingmetrics.csv"
)


def get_pseudoknots(y, min_distance_in_pk=6):
    y_pairs = struct_to_pairs(y)
    y_cogent_pairs = [(i + 1, j) for i, j in enumerate(y_pairs) if j > i + 1]

    def fast_pk_motif_search(cogent_pairs):
        left_links = {}
        for idx, (i, j) in enumerate(cogent_pairs):
            for (k, l) in cogent_pairs[idx + 1 :]:
                if k > j:
                    break
                if l > j:  # such that i < k < j < l by construction, ie a pseudoknot
                    if (i, j) in left_links:
                        left_links[(i, j)].append((k, l))
                    else:
                        left_links[(i, j)] = [(k, l)]
        pks = []
        for left_p, this_right_bound in left_links.items():
            for left_bound, right_bound in pks:
                if right_bound == this_right_bound:
                    left_bound.append(left_p)
                    break
            else:
                pks.append(([left_p], this_right_bound))

        return pks

    def get_distance_in_pseudoknot(leftb, rightb):
        leftb = list(sorted(leftb, key=lambda x: x[0]))
        rightb = list(sorted(rightb, key=lambda x: x[0]))
        # left_distance = rightb[-1][0] - leftb[0][0] - len(leftb) - len(rightb) + 1
        left_distance = rightb[-1][0] - leftb[0][0] - 1

        leftb = list(sorted(leftb, key=lambda x: x[1]))
        rightb = list(sorted(rightb, key=lambda x: x[1]))
        # right_distance = rightb[-1][1] - leftb[0][1] - len(leftb) - len(rightb) + 1
        right_distance = rightb[-1][1] - leftb[0][1] - 1

        return left_distance + right_distance

    ref = fast_pk_motif_search(y_cogent_pairs)
    ref = [
        (leftb, rightb)
        for leftb, rightb in ref
        if get_distance_in_pseudoknot(leftb, rightb) >= min_distance_in_pk
    ]

    return ref


def format_frags(frags):
    return [
        [
            [int(x) for x in ("[" + f2)[1:-1].split(" ")]
            for f2 in ("[[" + f)[1:-1].split("[")[1:]
        ]
        for f in frags[1:-1].split("[[")[1:]
    ]


recoverable = []
lengths = []
connected = []
structs = []
all_frags = []
for _, row in df.iterrows():
    pks = get_pseudoknots(row.struct)
    frags = format_frags(row.frags)
    attribs = np.zeros(len(row.struct), dtype=int)
    for i, f in enumerate(frags):
        for start, end in f:
            attribs[start : end + 1] = i
    for leftb, rightb in pks:
        left_attribs = []
        right_attribs = []
        for i, j in leftb:
            left_attribs.append(
                attribs[i - 1] if attribs[i - 1] == attribs[j - 1] else None
            )
        for i, j in rightb:
            right_attribs.append(
                attribs[i - 1] if attribs[i - 1] == attribs[j - 1] else None
            )
        recov = set(left_attribs) & set(right_attribs)
        if None in recov:
            recov.remove(None)
        assert len(recov) <= 1
        max_start = max([pair[0] for pair in leftb])
        min_end = min([pair[1] for pair in rightb])

        lengths.append(min_end - max_start)
        structs.append(row.struct)
        all_frags.append(frags)

        if not recov:
            recoverable.append(False)
            connected.append(np.nan)
            continue

        recoverable.append(True)
        connected.append(len(set(attribs[max_start - 1 : min_end])) <= 1)

res = pd.DataFrame(
    {
        "length": lengths,
        "recoverable": recoverable,
        "connected": connected,
        "struct": structs,
        "frags": all_frags,
    }
)


def plot_struct(struct, frags, height=0.335):
    n = len(struct)
    pairs_nopk = struct_to_pairs(re.sub("[^\(\)\.]", ".", struct))
    pairs_pk = struct_to_pairs(re.sub("[\(\)]", ".", struct))

    def draw_pairs(pairs, color):
        for i, j in enumerate(pairs):
            i += 1
            if i < j:
                x = np.array([i, (3 * i + j) / 4, (i + j) / 2, (i + 3 * j) / 4, j])
                y = np.array(
                    [
                        0,
                        height * 0.8 * (j - i) / n,
                        height * (j - i) / n,
                        height * 0.8 * (j - i) / n,
                        0,
                    ]
                )
                X_Y_Spline = make_interp_spline(x, y)
                X_ = np.linspace(i, j, 100)
                Y_ = X_Y_Spline(X_)
                plt.plot(X_, Y_, color=color, linewidth=0.5)

    # Draw
    draw_pairs(pairs_nopk, "silver")
    draw_pairs(pairs_pk, "firebrick")

    # colors = list(mcolors.TABLEAU_COLORS.keys())
    # np.random.shuffle(colors)
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    for p, frag in enumerate(frags):
        for i, j in frag:
            plt.plot([i - 0.5, j + 0.5], [-0.005, -0.005], color=colors[p], linewidth=5)

    plt.xlim([1, n])
    plt.ylim([-0.1, 1])
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off
    plt.show()


res_disconnected = res[(res.recoverable) & (~res.fillna(True).connected)].copy()
res_disconnected["frag_len"] = res_disconnected.frags.apply(len)
res_disconnected.sort_values("frag_len", inplace=True)

for struct, frags in zip(res_disconnected.struct, res_disconnected.frags):
    plot_struct(struct, frags)
