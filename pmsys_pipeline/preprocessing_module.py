from typing import List, Tuple, Union
import pandas as pd
import numpy as np

from pmsys_pipeline.pipeline_structure import Transformer


def get_histogram(feature: np.array):
    pass


def strip_data(input_df: pd.DataFrame) -> pd.DataFrame:
    mask = input_df.replace(0, np.nan)
    f_idx = mask.first_valid_index()
    l_idx = mask.last_valid_index()
    return input_df.loc[f_idx:l_idx, :]


def ts_train_val_split(
    predictor_ts: Union[pd.DataFrame, np.array],
    dependent_ts: Union[pd.DataFrame, np.array],
    window_size: int,
    lag: int,
    validate_size: float,
) -> Tuple[np.array, np.array]:

    predictor_ts = np.array(predictor_ts, ndmin=2)

    predictor_batches = [
        predictor_ts[i : i + window_size, :]
        for i in range(0, len(predictor_ts) - window_size, lag)
    ]
    dependent_batches = [
        dependent_ts[i : i + window_size, :]
        for i in range(0, len(dependent_ts) - window_size, lag)
    ]
    train_batches = []
    val_batches = []
    for pred_batch, dep_batch in zip(predictor_batches, dependent_batches):
        val_batch = dep_batch[len(dep_batch) - validate_size :, :]
        train_batch = pred_batch[: len(dep_batch) - validate_size, :]
        train_batches.append(train_batch)
        val_batches.append(val_batch)
    return train_batches, val_batches


class TSTrainTestSplit(Transformer):
    def __init__(self, window_size: int, lag: int, validate_size: float):
        super().__init__(
            ts_train_val_split,
            {
                "window_size": window_size,
                "lag": lag,
                "validate_size": validate_size,
            },
        )


def normalise_data(input_df: pd.DataFrame) -> pd.DataFrame:
    return (input_df - input_df.min()) / (input_df.max() - input_df.min())


class Normalise(Transformer):
    def __init__(self):
        super().__init__(
            normalise_data,
            {},
        )
