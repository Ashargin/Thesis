from predict import mxfold2_predict, linearfold_predict, ufold_predict, divide_predict, \
                        divide_get_cuts, linearfold_get_cuts
from utils import run_preds

run_preds(divide_predict, 'resources/divide_motifs_400_mx_preds.csv',
                          kwargs={'max_length': 400,
                                  'cut_fnc': divide_get_cuts, # with motifs input format
                                  'predict_fnc': mxfold2_predict})
