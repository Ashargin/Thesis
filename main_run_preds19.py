from predict import mxfold2_predict, linearfold_predict, ufold_predict, divide_predict, \
                        divide_get_cuts, linearfold_get_cuts
from utils import run_preds

run_preds(divide_predict, 'resources/divide_cheat_200_lf_preds.csv', use_structs=True,
                          kwargs={'max_length': 200,
                                  'predict_fnc': linearfold_predict})
