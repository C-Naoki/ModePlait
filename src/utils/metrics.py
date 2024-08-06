from typing import Optional

import cdt
import igraph as ig
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def rmse(true: np.ndarray, pred: Optional[np.ndarray] = None) -> float:
    if pred is None:
        pred = 0.0 * true
    diff = true.flatten() - pred.flatten()
    return np.sqrt(np.nanmean(pow(diff, 2)))


def mae(true: np.ndarray, pred: Optional[np.ndarray] = None) -> float:
    if pred is None:
        pred = 0.0 * true
    diff = true.flatten() - pred.flatten()
    return np.nanmean(np.abs(diff))


def count_accuracy(B_true: np.ndarray, B_est: np.ndarray, is_sid: bool = True) -> dict[str, float]:
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimated graph, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError("B_est should take value in {0,1,-1}")
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError("undirected edge should only appear once")
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError("B_est should take value in {0,1}")

    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)

    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])

    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])

    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)

    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)

    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    sid = float(cdt.metrics.SID(target=B_true, pred=B_est))

    metrics = {"fdr": fdr, "tpr": tpr, "fpr": fpr, "shd": shd, "sid": sid, "nnz": pred_size}
    if is_sid:
        try:
            sid = float(cdt.metrics.SID(target=B_true, pred=B_est))
            metrics["sid"] = sid
        except Exception:
            pass
    return metrics


def eval_metrics_reg(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": np.sqrt(mean_squared_error(true, pred)),
        "mae": mean_absolute_error(true, pred),
    }


def _is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()
