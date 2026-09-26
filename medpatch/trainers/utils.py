import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_new(df):
    """AUROC/AUPRC for one label column. Returns np.nan when undefined.

    AUROC needs both classes present. AUPRC is technically defined with no
    positives but is meaningless (sklearn returns 0.0 and warns), so it is
    treated as undefined too -- otherwise a phenotype with zero positives
    contributes a spurious 0.0 to the reported mean.
    """
    y_truth = np.asarray(df['y_truth'])
    y_pred = np.asarray(df['y_pred'])

    if len(np.unique(y_truth)) < 2:
        return np.nan, np.nan

    auprc = average_precision_score(y_truth, y_pred)
    auroc = roc_auc_score(y_truth, y_pred)
    return auprc, auroc


def bootstraping_eval(df, num_iter):
    """Bootstrap the metrics by resampling with replacement.

    A resample can contain only one class even when the full column has both --
    likelier the rarer the phenotype -- so those draws are discarded. The count
    is returned so the caller can report how degraded the interval is.
    """
    auroc_list = []
    auprc_list = []
    discarded = 0

    for _ in range(num_iter):
        sample = df.sample(frac=1, replace=True)
        auprc, auroc = evaluate_new(sample)
        # `auroc != np.nan` is ALWAYS True (IEEE 754: nan != nan), so the
        # original filter never rejected anything. NaNs entered the lists and
        # np.percentile then returned NaN, so confidence intervals looked
        # computed but were not.
        if np.isnan(auprc) or np.isnan(auroc):
            discarded += 1
            continue
        auroc_list.append(auroc)
        auprc_list.append(auprc)

    return auprc_list, auroc_list, discarded


def computing_confidence_intervals(list_, true_value):
    """95% CI from the bootstrap distribution. Returns (nan, nan) if undefined.

    `true_value - list_` used to be evaluated with `list_` a plain Python list.
    That only worked because sklearn returns numpy scalars, which broadcast over
    a list. When the metric was undefined the value was `np.nan`, a plain Python
    float, and the subtraction raised
        TypeError: unsupported operand type(s) for -: 'float' and 'list'
    """
    values = np.asarray(list_, dtype=float)

    if np.isnan(true_value) or values.size == 0:
        return (np.nan, np.nan)

    delta = true_value - values
    delta_lower = np.percentile(delta, 97.5)
    delta_upper = np.percentile(delta, 2.5)

    upper = true_value - delta_upper
    lower = true_value - delta_lower
    return (upper, lower)


def get_model_performance(df, label=None, num_iter=1000, verbose=True):
    """Point estimates plus 95% CIs for one label column.

    `label` names the column in warnings, so an undefined metric can be traced
    to a specific phenotype rather than a bare index.
    """
    test_auprc, test_auroc = evaluate_new(df)
    name = label if label is not None else 'column'

    if np.isnan(test_auroc):
        n_pos = int(np.asarray(df['y_truth']).sum())
        if verbose:
            print(f"  [metrics] {name}: AUROC/AUPRC undefined "
                  f"({n_pos} positives out of {len(df)}) -- reporting (nan, nan). "
                  "Not an error: the metric needs both classes present.")
        return (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan)

    auprc_list, auroc_list, discarded = bootstraping_eval(df, num_iter=num_iter)

    if verbose and discarded:
        print(f"  [metrics] {name}: {discarded}/{num_iter} bootstrap samples "
              f"discarded (resample had only one class); CI computed from "
              f"{len(auroc_list)} samples.")

    upper_auprc, lower_auprc = computing_confidence_intervals(auprc_list, test_auprc)
    upper_auroc, lower_auroc = computing_confidence_intervals(auroc_list, test_auroc)

    return ((test_auprc, upper_auprc, lower_auprc),
            (test_auroc, upper_auroc, lower_auroc))
