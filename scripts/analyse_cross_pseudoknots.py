import pandas as pd
import itertools

from src.utils import struct_to_pairs

df = pd.read_csv(rf"resources/results/16S23S_pk.csv")
df = df[df.model == "DivideFold MLP (1000) + KnotFold"]


def get_pseudoknot_motif_scores(y):
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

    motifs = fast_pk_motif_search(y_cogent_pairs)

    # Evaluate "cross" pseudoknots
    distances = []
    for leftb, rightb in motifs:
        leftb = list(sorted(leftb, key=lambda x: x[0]))
        rightb = list(sorted(rightb, key=lambda x: x[0]))
        # left_distance = rightb[-1][0] - leftb[0][0] - len(leftb) - len(rightb) + 1
        left_distance = rightb[-1][0] - leftb[0][0] - 1

        leftb = list(sorted(leftb, key=lambda x: x[1]))
        rightb = list(sorted(rightb, key=lambda x: x[1]))
        # right_distance = rightb[-1][1] - leftb[0][1] - len(leftb) - len(rightb) + 1
        right_distance = rightb[-1][1] - leftb[0][1] - 1

        distances.append(left_distance + right_distance)

    return distances


ref_distances = df.struct.apply(get_pseudoknot_motif_scores)
pred_distances = df.pred.apply(get_pseudoknot_motif_scores)
ref_distances = pd.Series(sorted(itertools.chain.from_iterable(ref_distances)))
pred_distances = pd.Series(sorted(itertools.chain.from_iterable(pred_distances)))

print(ref_distances.value_counts().sort_index().cumsum()[:50] / len(ref_distances))
print(pred_distances.value_counts().sort_index().cumsum()[:50] / len(pred_distances))

import matplotlib.pyplot as plt
import seaborn as sns

sns.kdeplot(ref_distances, label="Reference")
sns.kdeplot(pred_distances, label="Predicted")
plt.show()
