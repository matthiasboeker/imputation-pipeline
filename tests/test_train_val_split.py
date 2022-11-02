import numpy as np
import pandas as pd

from pmsys_pipeline.preprocessing_module import TSTrainValSplit

input_ts = pd.DataFrame(
    {
        "feature_a": np.arange(1, 20),
        "feature_b": np.arange(1, 20),
        "feature_c": np.arange(1, 20),
    }
)
y_ts = np.array(input_ts["feature_a"]).reshape(-1, 1)


def test_train_val_split():
    first_split = TSTrainValSplit(window_size=10, lag=5, validation_size=1)
    second_split = TSTrainValSplit(window_size=10, lag=0, validation_size=1)
    third_split = TSTrainValSplit(window_size=10, lag=10, validation_size=1)
    fourth_split = TSTrainValSplit(window_size=10, lag=5, validation_size=10)
    fifth_split = TSTrainValSplit(window_size=10, lag=5, validation_size=0)

    first_train, first_val = first_split.fit_transform(input_ts, y_ts)
    print(first_train)
    print(first_val)
    second_train, second_val = first_split.fit_transform(input_ts, y_ts)
    third_train, third_val = first_split.fit_transform(input_ts, y_ts)
    fourth_train, fourth_val = first_split.fit_transform(input_ts, y_ts)
    fifth_train, fifth_val = first_split.fit_transform(input_ts, y_ts)
