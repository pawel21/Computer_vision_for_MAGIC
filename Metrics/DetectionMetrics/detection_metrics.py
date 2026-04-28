import numpy as np
from dataclasses import dataclass


@dataclass
class DetectionMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    iou: float

    @property
    def summary(self) -> str:
        return (f"P={self.precision:.3f} R={self.recall:.3f} "
                f"F1={self.f1:.3f} IoU={self.iou:.3f} "
                f"(TP={self.tp}, FP={self.fp}, FN={self.fn})")


def aggregate_outliers(
        outliers: dict,
        strategy: str = 'union',
        min_votes: int = 2,
        n_mirrors: int = 249,
) -> np.ndarray:
    """
    Łączy wyniki z wszystkich features w jedną listę podejrzanych luster.

    strategy:
        'union'  — lustro outlier jeśli wykryte przez jakikolwiek feature
        'voting' — lustro outlier jeśli wykryte przez >= min_votes features

    Returns: array indeksów luster oznaczonych jako outliers
    """
    votes = np.zeros(n_mirrors, dtype=int)
    for key, res in outliers.items():
        flagged = np.concatenate([res['high'], res['low']])
        votes[flagged] += 1

    if strategy == 'union':
        return np.where(votes >= 1)[0]
    elif strategy == 'voting':
        return np.where(votes >= min_votes)[0]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")