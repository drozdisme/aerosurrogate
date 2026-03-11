"""Train/validation/test splitting for AeroSurrogate."""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Container for train/val/test splits."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    target_names: List[str]


def split_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> DataSplit:
    """Split DataFrame into train/validation/test sets.

    Ensures no data leakage by performing a two-stage stratified-ish split.
    """
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    # Second split: separate validation from train
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, random_state=random_state,
    )

    logger.info(
        f"Split complete: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    return DataSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_cols,
        target_names=target_cols,
    )
