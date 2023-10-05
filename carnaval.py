import sys

sys.path.append("carnaval_dataset")
import RIN

import pickle
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Open carnaval dict # graphs
# f = open(r'carnaval_dataset\graphs_2.92_nx3_with_SSEs.pickle', 'rb')
# carnaval_dict = pickle.load(f)
# f.close()

# Open carnaval dict # RINs
f = open(r"carnaval_dataset\CaRNAval_1_as_dictionnary.nxpickled", "rb")
carnaval_dict = pickle.load(f)
f.close()
rins = [carnaval_dict[i + 1] for i in range(len(carnaval_dict))]


def graph_to_motif_seq(g):
    # Get components
    nodes = np.array(sorted(list(g)))
    new_component = np.append(True, nodes[1:] != nodes[:-1] + 1)
    component_idx_start = np.argwhere(new_component).ravel()
    component_idx_end = np.append(component_idx_start[1:], len(nodes))
    components = [
        nodes[start:end].tolist()
        for start, end in zip(component_idx_start, component_idx_end)
    ]

    motif_seq = "*".join(
        ["".join([g.nodes[n]["nt"] for n in comp]) for comp in components]
    )

    return motif_seq


motif_seqs = pd.Series(
    [graph_to_motif_seq(rin.graph) for rin in rins], index=[rin.ID for rin in rins]
).drop_duplicates()
motif_seqs.name = "motif_seq"
motif_seqs.to_csv(r"resources\motif_seqs.csv")
