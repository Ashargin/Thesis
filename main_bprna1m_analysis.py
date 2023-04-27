import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dbn_path = os.path.join('bpRNA_1m', 'dbnFiles')
letters = {'A', 'C', 'G', 'U', 'R', 'Y', 'S', 'W', 'M', 'K', 'B', 'N', 'V', 'H', 'S', 'D', 'O', 'P', 'X', 'I', 'T', '.', '_', '~'}
brackets = {'.', '(', ')', '[', ']', '<', '>', '{', '}', 'a', 'A', 'b', 'B', 'c', 'C'}
lens = []
letter_counts = {}
bracket_counts = {}
all_dbn_path = os.path.join(dbn_path, 'allDbn.dbn')
f_out = open(all_dbn_path, 'w')
for i, filename in enumerate(os.listdir(dbn_path)):
    if i % 10000 == 0:
        print(i)
    if filename.startswith('bpRNA'):
        file_path = os.path.join(dbn_path, filename)
        with open(file_path, 'r') as f:
            lines = f.read().split('\n')
            assert all([l.startswith('#') for l in lines[:3]]) \
                and set(lines[3].upper()).issubset(letters) \
                and set(lines[4]).issubset(brackets) \
                and len(lines[3]) == len(lines[4]) \
                and (len(lines) == 5
                        or len(lines) == 6 and lines[5] == '')
            name, seq, struct = lines[0], lines[3], lines[4]
            if len(seq) == 0:
                continue
            lens.append(len(seq))
            for l in set(seq):
                if l in letter_counts:
                    letter_counts[l] += seq.count(l)
                else:
                    letter_counts[l] = seq.count(l)
            for b in set(struct):
                if b in bracket_counts:
                    bracket_counts[l] += struct.count(b)
                else:
                    bracket_counts[l] = struct.count(b)
            if filename != 'bpRNA_CRW_1.dbn':
                f_out.write('\n')
            f_out.write(name)
            f_out.write('\n' + seq)
            f_out.write('\n' + struct)
            f.close()
f_out.close()

sns.kdeplot(lens, fill=True, linewidth=2)
plt.show()


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


n = 501
X = np.arange(len(lens))
y = np.array(lens)
plt.plot(X[int((n-1)/2):-int((n-1)/2)], moving_average(y, n=n))
