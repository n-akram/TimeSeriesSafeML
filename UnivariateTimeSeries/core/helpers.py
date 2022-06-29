import numpy as np
import pandas as pd

def compute_difference(gt, preds):

    """This function computes absolute difference between ground truth and predictions.

    :param array gt: Ground truth
    :param array preds: Predictions
    :return numpy.ndarray: Difference values
    """    
    gt_preds_df = pd.DataFrame({
        'GT': gt,
        'Predictions': preds
    })

    gt_preds_df['diff'] = (gt_preds_df['GT'] - gt_preds_df['Predictions']).abs()

    return gt_preds_df['diff'].values