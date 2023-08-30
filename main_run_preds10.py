from predict import mxfold2_predict, linearfold_predict, ufold_predict, divide_predict, \
                        divide_get_cuts, linearfold_get_cuts
from utils import run_preds

run_preds(divide_predict, 'resources/divide_cheat_600_mx_preds.csv', use_structs=True,
                          kwargs={'max_length': 600,
                                  'predict_fnc': mxfold2_predict})
