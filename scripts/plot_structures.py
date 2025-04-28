import numpy as np
import re
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

from src.utils import struct_to_pairs


def plot_struct(struct, height=0.335):
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
                plt.plot(X_, Y_, color=color, linewidth=2.0)

    # Draw
    draw_pairs(pairs_nopk, "silver")
    draw_pairs(pairs_pk, "firebrick")

    plt.xlim([1, n])
    plt.ylim([0.0, 1])
    plt.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )  # labels along the bottom edge are off
    plt.show()


plot_struct(".(..)")

# struct = ".....((((((((...........[[[......................(((((...........................((((.....[[[......((((((((.......(((((..]]]...))))).....))))))))......(((((...(((((...((((((...(((...[[[..)))......))))))...)....]]]...((....)).))))..))).))........))))....((((((....(...((((.....[[[...))))...).))))))..........(((((..((((((((..]]].....[[[..))))))))...((..))..]]]..)))))...............................]]]...............)))))............))))))))......"
# plot_struct(struct)
# plot_struct(re.sub("[^\(\)\.]", ".", struct))
#
# struct = ".....((((((((...........[[[......................(((((.................................]]]...............)))))............))))))))......"
# plot_struct(struct)
# plot_struct(re.sub("[^\(\)\.]", ".", struct))
#
# struct = "...........((((.....[[[......((((((((.......(((((..]]]...))))).....))))))))......(((((...(((((...((((((...(((...[[[..)))......))))))...)....]]]...((....)).))))..))).))........)))).."
# plot_struct(struct)
# plot_struct(re.sub("[^\(\)\.]", ".", struct))
#
# struct = "..((((((....(...((((...........))))...).))))))....."
# plot_struct(struct)
# plot_struct(re.sub("[^\(\)\.]", ".", struct))
#
# struct = (
#     ".....(((((..((((((((..........[[[..))))))))...((..))..]]]..)))))..............."
# )
# plot_struct(struct)
# plot_struct(re.sub("[^\(\)\.]", ".", struct))
