import os
import numpy as np
from utils import pairs_to_struct

bpseq_paths = [r'rnapar_raw_data/ArchiveII',
               r'rnapar_raw_data/bpRNA',
               r'rnapar_raw_data/PDB/bpseq',
               r'rnapar_raw_data/RNAStrAlign',
               ]
dp_paths = [r'rnapar_raw_data/RMDB',
             r'rnapar_raw_data/RNAStrand',
             ]


def read_bpseq_from_folders(paths):
    res_txt = ''

    for j, p in enumerate(paths):
        files = os.listdir(p)
        for i, f in enumerate(files):
            print(f'Folder {j+1}/{len(paths)}, file {i+1}/{len(files)}')
            name, _ = os.path.splitext(f)
            txt = open(os.path.join(p, f), 'r').read()

            lines = txt.replace('\t', ' ').split('\n')
            lines = [l for l in lines if l and l[0].isdigit()]
            assert all([l.count(' ') == 2 for l in lines])
            n_nc = len(lines)
            elements = ' '.join(lines).split(' ')

            source, nc, target = elements[0::3], elements[1::3], elements[2::3]
            source = [int(s) for s in source]
            nc = [nuc.replace('T', 'U') if nuc in ['A', 'U', 'C', 'T', 'G'] else 'N'
                                                                    for nuc in nc]
            target = [int(t) for t in target]
            target = [0 if t == -1 else t for t in target]
            assert set(nc).issubset(set(['A', 'U', 'C', 'G', 'N']))
            assert set(source) == set(range(1, n_nc + 1))
            assert set(target).issubset(set(range(n_nc + 1)))
            for l in range(1, n_nc + 1):
                if target[l - 1] != 0:
                    assert target[target[l - 1] - 1] == l

            seq = ''.join(nc)
            struct = pairs_to_struct(np.array(target))
            pairing_chars = ['.', '(', ')', '[', ']', '{', '}', '<', '>']
            assert len(struct) == len(seq)
            assert not struct[0].isalpha() or struct[0].islower()
            assert set(struct).issubset(set(pairing_chars))
            for c_open, c_close in zip(pairing_chars[1::2], pairing_chars[2::2]):
                assert struct.count(c_open) == struct.count(c_close)

            res_txt += f'#Name: {name}\n{seq}\n{struct}\n'

    return res_txt


def read_dp_from_folders(paths):
    res_txt = ''

    for j, p in enumerate(paths):
        files = os.listdir(p)
        for i, f in enumerate(files):
            print(f'Folder {j+1}/{len(paths)}, file {i+1}/{len(files)}')
            name, _ = os.path.splitext(f)
            txt = open(os.path.join(p, f), 'r').read()

            txt = txt.strip()
            if not '\n' in txt:
                continue
            assert txt.count('\n') == 1

            seq, struct = txt.split('\n')
            if len(seq) != len(struct):
                continue

            seq = seq.upper()
            seq = seq.replace('T', 'U').replace('X', 'N')
            assert set(seq).issubset(set(['A', 'U', 'C', 'G', 'N']))
            pairing_chars = ['.', '(', ')', '[', ']', '{', '}', '<', '>']
            for i in range(65, 91):
                pairing_chars += [chr(i).lower(), chr(i)]
            assert not struct[0].isalpha() or struct[0].islower()
            assert set(struct).issubset(set(pairing_chars))
            for c_open, c_close in zip(pairing_chars[1::2], pairing_chars[2::2]):
                if not struct.count(c_open) == struct.count(c_close):
                    break

            else:
                res_txt += f'#Name: {name}\n{seq}\n{struct}\n'

    return res_txt


txt_from_bpseq = read_bpseq_from_folders(bpseq_paths)
txt_from_dp = read_dp_from_folders(dp_paths)
with open('rnapar_raw_data/allDbn.dbn', 'w') as f:
    f.write(txt_from_bpseq + txt_from_dp)
