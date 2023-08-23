import platform
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if platform.system() == 'Linux':
    from predict import mxfold2_predict, linearfold_predict, ufold_predict, divide_predict, \
                        divide_get_cuts, linearfold_get_cuts
from utils import run_preds, get_scores_df

## Settings
path_mxfold2_preds = 'resources/mxfold2_preds.csv'
path_linearfold_preds = 'resources/linearfold_preds.csv'
path_ufold_preds = 'resources/ufold_preds.csv'
path_divide_cheat_preds = 'resources/divide_cheat_preds.csv'
path_divide_dnabert_preds = 'resources/divide_dnabert_preds.csv'
path_divide_motifs_preds = 'resources/divide_motifs_preds.csv'
path_divide_linearfoldcuts_preds = 'resources/divide_linearfoldcuts_preds.csv'
path_divide_motifs_stop385_preds = 'resources/divide_motifs_stop385_preds.csv'

## Run mxfold2
if platform.system() == 'Linux':
    run_preds(mxfold2_predict, path_mxfold2_preds, allow_errors=True)

## Run linearfold
if platform.system() == 'Linux':
    run_preds(linearfold_predict, path_linearfold_preds, allow_errors=True)

## Run ufold
# if platform.system() == 'Linux':
    # run_preds(ufold_predict, path_ufold_preds, allow_errors=True)

## Run divide cheat
if platform.system() == 'Linux':
    run_preds(divide_predict, path_divide_cheat_preds, use_structs=True,
                              kwargs={'max_length': 200,
                                      'predict_fnc': mxfold2_predict})

## Run divide
# if platform.system() == 'Linux':
#     run_preds(divide_predict, path_divide_dnabert_preds, max_len=2560,
#                               kwargs={'max_length': 200,
#                                       'cut_fnc': divide_get_cuts, # with dnabert input format
#                                       'predict_fnc': mxfold2_predict})

if platform.system() == 'Linux':
    run_preds(divide_predict, path_divide_motifs_preds,
                              kwargs={'max_length': 200,
                                      'cut_fnc': divide_get_cuts, # with motifs input format
                                      'predict_fnc': mxfold2_predict})

if platform.system() == 'Linux':
    run_preds(divide_predict, path_divide_linearfoldcuts_preds,
                              kwargs={'max_length': 200,
                                      'cut_fnc': linearfold_get_cuts,
                                      'predict_fnc': mxfold2_predict})

if platform.system() == 'Linux':
    run_preds(divide_predict, path_divide_motifs_stop385_preds,
                              kwargs={'max_length': 385,
                                      'cut_fnc': divide_get_cuts, # with motifs input format
                                      'predict_fnc': mxfold2_predict})

## Score predictions
mxfold2_scores = get_scores_df(path_mxfold2_preds)
linearfold_scores = get_scores_df(path_linearfold_preds)
# ufold_scores = get_scores_df(path_ufold_preds)
divide_cheat_scores = get_scores_df(path_divide_cheat_preds)
# divide_dnabert_scores = get_scores_df(path_divide_dnabert_preds)
divide_motifs_scores = get_scores_df(path_divide_motifs_preds)
divide_linearfoldcuts_scores = get_scores_df(path_divide_linearfoldcuts_preds)

mxfold2_scores['model'] = 'mxfold2'
linearfold_scores['model'] = 'linearfold'
# ufold_scores['model'] = 'ufold'
divide_cheat_scores['model'] = 'recursive_strategy_from_truth'
# divide_dnabert_scores['model'] = 'divide_dnabert'
divide_motifs_scores['model'] = 'recursive_predicted_from_motifs'
divide_linearfoldcuts_scores['model'] = 'divide_strategy_from_linearfold'
data = pd.concat([mxfold2_scores, linearfold_scores, divide_cheat_scores,
                  divide_motifs_scores, divide_linearfoldcuts_scores])

## Plot scores
plt.figure()
sns.kdeplot(data=data, x='length', y='ppv', weights='weight', hue='model', fill=True, alpha=.5)
plt.xlabel('Sequence length')
plt.ylabel('PPV')
plt.title('PPV vs sequence length')

plt.figure()
sns.kdeplot(data=data, x='length', y='sen', weights='weight', hue='model', fill=True, alpha=.5)
plt.xlabel('Sequence length')
plt.ylabel('SEN')
plt.title('SEN vs sequence length')

plt.figure()
sns.kdeplot(data=data, x='length', y='fscore', weights='weight', hue='model', fill=True, alpha=.5, thresh=0.32)
plt.xlabel('Sequence length')
plt.ylabel('F-score')
plt.title('F-score vs sequence length')

## Plot time and memory constraints
def round_lengths(df, n=10):
    df = df.copy()
    df.length = df.length.apply(lambda x: round(x / n) * n)
    return df


plt.figure()
sns.lineplot(data=round_lengths(data, n=30), x='length', y='time', hue='model')
plt.xlabel('Sequence length')
plt.ylabel('Time (s)')
plt.title('Computation time vs sequence length')

plt.figure()
sns.lineplot(data=round_lengths(data, n=30), x='length', y='memory', hue='model')
plt.xlabel('Sequence length')
plt.ylabel('Memory consumption (fraction of total memory)')
plt.title('Memory consumption vs sequence length')
plt.show()
