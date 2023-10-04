import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_scores_df

## Score predictions
# mxfold2_scores = get_scores_df('resources/mxfold2_preds.csv')
# linearfold_scores = get_scores_df('resources/linearfold_preds.csv')
# ufold_scores = get_scores_df('resources/ufold_preds.csv')
# divide_cheat_200_lf_scores = get_scores_df('resources/divide_cheat_200_lf_preds.csv')
# divide_cheat_200_mx_scores = get_scores_df('resources/divide_cheat_200_mx_preds.csv')
# divide_cheat_400_mx_scores = get_scores_df('resources/divide_cheat_400_mx_preds.csv')
# divide_cheat_600_mx_scores = get_scores_df('resources/divide_cheat_600_mx_preds.csv')
# divide_cheat_800_mx_scores = get_scores_df('resources/divide_cheat_800_mx_preds.csv')
# divide_cheat_1000_mx_scores = get_scores_df('resources/divide_cheat_1000_mx_preds.csv')
# divide_linearfoldcuts_200_lf_scores = get_scores_df('resources/divide_linearfoldcuts_200_lf_preds.csv')
# divide_linearfoldcuts_200_mx_scores = get_scores_df('resources/divide_linearfoldcuts_200_mx_preds.csv')
# divide_linearfoldcuts_400_mx_scores = get_scores_df('resources/divide_linearfoldcuts_400_mx_preds.csv')
# divide_linearfoldcuts_600_mx_scores = get_scores_df('resources/divide_linearfoldcuts_600_mx_preds.csv')
# divide_linearfoldcuts_800_mx_scores = get_scores_df('resources/divide_linearfoldcuts_800_mx_preds.csv')
# divide_linearfoldcuts_1000_mx_scores = get_scores_df('resources/divide_linearfoldcuts_1000_mx_preds.csv')
# divide_motifs_200_mx_scores = get_scores_df('resources/divide_motifs_200_mx_preds.csv')
# divide_motifs_400_mx_scores = get_scores_df('resources/divide_motifs_400_mx_preds.csv')
# divide_motifs_600_mx_scores = get_scores_df('resources/divide_motifs_600_mx_preds.csv')
# divide_motifs_800_mx_scores = get_scores_df('resources/divide_motifs_800_mx_preds.csv')
# divide_motifs_1000_mx_scores = get_scores_df('resources/divide_motifs_1000_mx_preds.csv')
# divide_motifs_1step_mx_scores = get_scores_df('resources/divide_motifs_1step_mx_preds.csv')
# divide_motifs_2step_mx_scores = get_scores_df('resources/divide_motifs_2step_mx_preds.csv')
# divide_motifs_3step_mx_scores = get_scores_df('resources/divide_motifs_3step_mx_preds.csv')
# divide_motifs_4step_mx_scores = get_scores_df('resources/divide_motifs_4step_mx_preds.csv')
# divide_motifs_5step_mx_scores = get_scores_df('resources/divide_motifs_5step_mx_preds.csv')
#
# mxfold2_scores['model'] = 'MXfold2'
# linearfold_scores['model'] = 'LinearFold'
# ufold_scores['model'] = 'UFold'
# divide_cheat_200_lf_scores['model'] = 'Recursive strategy from true structure (until 200 nc, with LinearFold as structure prediction model)'
# divide_cheat_200_mx_scores['model'] = 'Recursive strategy from true structure (until 200 nc)'
# divide_cheat_400_mx_scores['model'] = 'Recursive strategy from true structure (until 400 nc)'
# divide_cheat_600_mx_scores['model'] = 'Recursive strategy from true structure (until 600 nc)'
# divide_cheat_800_mx_scores['model'] = 'Recursive strategy from true structure (until 800 nc)'
# divide_cheat_1000_mx_scores['model'] = 'Recursive strategy from true structure (until 1000 nc)'
# divide_linearfoldcuts_200_lf_scores['model'] = 'Recursive strategy from LinearFold preds (until 200 nc, with LinearFold as structure prediction model)'
# divide_linearfoldcuts_200_mx_scores['model'] = 'Recursive strategy from LinearFold preds (until 200 nc)'
# divide_linearfoldcuts_400_mx_scores['model'] = 'Recursive strategy from LinearFold preds (until 400 nc)'
# divide_linearfoldcuts_600_mx_scores['model'] = 'Recursive strategy from LinearFold preds (until 600 nc)'
# divide_linearfoldcuts_800_mx_scores['model'] = 'Recursive strategy from LinearFold preds (until 800 nc)'
# divide_linearfoldcuts_1000_mx_scores['model'] = 'Recursive strategy from LinearFold preds (until 1000 nc)'
# divide_motifs_200_mx_scores['model'] = 'Our approach using predicted cut points (until 200 nc)'
# divide_motifs_400_mx_scores['model'] = 'Until 400 nc'
# divide_motifs_600_mx_scores['model'] = 'Until 600 nc'
# divide_motifs_800_mx_scores['model'] = 'Until 800 nc'
# divide_motifs_1000_mx_scores['model'] = 'Until 1000 nc'
# divide_motifs_1step_mx_scores['model'] = 'Our approach using predicted cut pointsl (limited to 1 step)'
# divide_motifs_2step_mx_scores['model'] = 'Limited to 2 steps'
# divide_motifs_3step_mx_scores['model'] = 'Limited to 3 steps'
# divide_motifs_4step_mx_scores['model'] = 'Limited to 4 steps'
# divide_motifs_5step_mx_scores['model'] = 'Limited to 5 steps'
#
# data = pd.concat([mxfold2_scores, linearfold_scores, ufold_scores,
#                   divide_cheat_200_lf_scores, divide_cheat_200_mx_scores,
#                   divide_cheat_400_mx_scores, divide_cheat_600_mx_scores,
#                   divide_cheat_800_mx_scores, divide_cheat_1000_mx_scores,
#                   divide_linearfoldcuts_200_lf_scores, divide_linearfoldcuts_200_mx_scores,
#                   divide_linearfoldcuts_400_mx_scores, divide_linearfoldcuts_600_mx_scores,
#                   divide_linearfoldcuts_800_mx_scores, divide_linearfoldcuts_1000_mx_scores,
#                   divide_motifs_200_mx_scores, divide_motifs_400_mx_scores,
#                   divide_motifs_600_mx_scores, divide_motifs_800_mx_scores,
#                   divide_motifs_1000_mx_scores])
# data.reset_index(inplace=True, drop=True)
# data.to_csv(r'resources\all_results.csv')

data = pd.read_csv(r'resources\all_results.csv', index_col=0)
data = data[data.fscore.notna()]
data = data[(data.length < 1650) | ((data.length > 2750) & (data.length < 3800))]
rna_names_all_models = data.groupby('rna_name').model.nunique() == data.model.nunique()
data = data[(data.rna_name.isin(rna_names_all_models[rna_names_all_models].index)) | (data.length >= 1000)]

data_200 = data[(data.model.str.contains('200')) | (data.model.isin(['MXfold2', 'LinearFold', 'UFold']))]
data_400 = data[(data.model.str.contains('400')) | (data.model.isin(['MXfold2', 'LinearFold', 'UFold']))]
data_600 = data[(data.model.str.contains('600')) | (data.model.isin(['MXfold2', 'LinearFold', 'UFold']))]
data_800 = data[(data.model.str.contains('800')) | (data.model.isin(['MXfold2', 'LinearFold', 'UFold']))]
data_1000 = data[(data.model.str.contains('1000')) | (data.model.isin(['MXfold2', 'LinearFold', 'UFold']))]
data_true = data[data.model.str.contains('true')]
data_linearfoldcuts = data[data.model.str.contains('LinearFold preds')]
data_motifs = data[data.model.str.contains('Our approach')]

## Plot scores
def round_lengths(df, n1=200, n2=400):
    df = df.copy()
    df.length = df.length.apply(lambda x: round(x / n1) * n1 if x < 1000 else round(x / n2) * n2)
    return df

plt.figure()
sns.lineplot(data=round_lengths(data), x='length', y='fscore', hue='model', estimator='mean', palette=['tab:green', 'firebrick', 'orangered', 'orange', 'gold', 'lightgoldenrodyellow'])
plt.xlabel('Sequence length')
plt.ylabel('F-score')
plt.title('F-score vs sequence length')
plt.ylim([0.25, 0.85])
plt.show()

## Plot time and memory constraints
plt.figure()
sns.lineplot(data=round_lengths(data, n1=50, n2=50), x='length', y='time', hue='model', estimator='mean', palette=['tab:blue', 'tab:orange', 'tab:olive', 'firebrick'])
plt.xlabel('Sequence length')
plt.ylabel('Time (s)')
plt.title('Computation time vs sequence length')
plt.show()

plt.figure()
sns.lineplot(data=round_lengths(data, n1=50, n2=50), x='length', y='memory', hue='model', estimator='mean', palette=['tab:blue', 'tab:orange', 'tab:olive', 'firebrick'])
plt.xlabel('Sequence length')
plt.ylabel('Memory cost / total memory')
plt.title('Memory cost vs sequence length')
plt.show()
