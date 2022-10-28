from typing import Tuple, Union
import pandas as pd
import numpy as np

from pmsys_pipeline.pipeline_structure import Transformer


def strip_data(input_df: pd.DataFrame) -> pd.DataFrame:
    mask = input_df.replace(0, np.nan)
    f_idx = mask.first_valid_index()
    l_idx = mask.last_valid_index()
    return input_df.loc[f_idx:l_idx, :]


def split_ts():
    pass


@Transformer
def forward_fill_data(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df.ffill()


@Transformer
def normalise_data(input_df: pd.DataFrame) -> pd.DataFrame:
    return (input_df - input_df.min()) / (input_df.max() - input_df.min())


@Transformer
def ts_train_val_split(
    predictor_ts: Union[pd.DataFrame, np.array],
    dependent_ts: Union[pd.Series, np.array, None],
    window_size: int,
    lag: int,
    validate_size: float,
) -> Tuple[np.array, np.array]:

    predictor_ts = np.array(predictor_ts, ndmin=2).T
    if dependent_ts is None:
        dependent_ts = predictor_ts
    predictor_batches = [predictor_ts[i: i + window_size, :] for i in range(0, len(predictor_ts)-window_size, lag)]
    dependent_batches = [dependent_ts[i: i + window_size, :] for i in range(0, len(dependent_ts)-window_size, lag)]
    train_batches: list = []
    val_batches: list = []
    for pred_batch, dep_batch in zip(predictor_batches, dependent_batches):
        val_batch = dep_batch[len(dep_batch)-validate_size:, :]
        train_batch = pred_batch[:len(dep_batch)-validate_size, :]
        train_batches.append(train_batch)
        val_batches.append(val_batch)
    train_tensor = np.dstack(train_batches)
    val_tensor = np.dstack(val_batches)
    return train_tensor, val_tensor





