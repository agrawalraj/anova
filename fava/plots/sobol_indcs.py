
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm

import seaborn as sns
import pandas as pd
from fava.plots.colors import red_rgb, blue_rgb

sns.set_style('whitegrid')


def make_default_name(cov_ix):
    return f'Feature {cov_ix}'


def get_covariate_variation(X, decomposer):
    covariate_importance = dict()
    for cov_ix in tqdm(decomposer.selected_covs):
        covariate_importance[cov_ix] = decomposer.get_variation_at_covariate(X, cov_ix, verbose=False).var().item()
    return covariate_importance


def sobol_importance(X, decomposer, ax=None, feature_names=None, max_feats=10):
    covariate_importance = get_covariate_variation(X, decomposer)
    sobol_index_df = pd.DataFrame(list(covariate_importance.items()))
    sobol_index_df.columns = ['cov_ix', 'variance']
    
    if feature_names is None:
        sobol_index_df['names'] = sobol_index_df['cov_ix'].apply(make_default_name)
    else:
        assert len(feature_names) == X.shape[1]
        sobol_index_df['names'] = sobol_index_df['cov_ix'].apply(lambda x: feature_names[x])
    
    sobol_index_df = sobol_index_df.sort_values(by='variance', ascending=False)
    sobol_index_df['norm_variance'] = 100 * sobol_index_df['variance'].values / sobol_index_df['variance'].sum()

    if sobol_index_df.shape[0] > max_feats:
        top_feats = sobol_index_df.head(max_feats)
        n_removed_feats = sobol_index_df.shape[0] - max_feats
        remaining_feats = sobol_index_df.tail(n_removed_feats)
        remaining_variance = remaining_feats['norm_variance'].sum()
        remain_df = pd.DataFrame({'names': [f'{n_removed_feats} other features'], 'norm_variance': [remaining_variance]})
        top_feats = pd.concat([top_feats, remain_df])

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    sns.barplot(x='norm_variance', y='names', data=top_feats, color=blue_rgb, ax=ax)
    plt.xlabel('Proportion of Variance Explained')
    plt.ylabel('')
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    sns.despine()
    plt.tight_layout()
    return covariate_importance
