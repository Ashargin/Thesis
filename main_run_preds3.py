from predict import mxfold2_predict, linearfold_predict, ufold_predict, divide_predict, \
                        divide_get_cuts, linearfold_get_cuts
from utils import run_preds

run_preds(linearfold_predict, 'resources/linearfold_preds.csv', allow_errors=True)
