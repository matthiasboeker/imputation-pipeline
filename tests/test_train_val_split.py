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

output_first_train = [
    pd.DataFrame(
        np.array(
            [
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
                [6, 6, 6],
                [7, 7, 7],
            ]
        ),
        columns=["feature_a", "feature_b", "feature_c"],
    ),
    pd.DataFrame(
        np.array(
            [
                [6, 6, 6],
                [7, 7, 7],
                [8, 8, 8],
                [9, 9, 9],
                [10, 10, 10],
                [11, 11, 11],
                [12, 12, 12],
            ]
        ),
        columns=["feature_a", "feature_b", "feature_c"],
    ),
]

output_first_val = [pd.DataFrame([[8], [9], [10]]), pd.DataFrame([[13], [14], [15]])]

output_second_train = [
    pd.DataFrame(
        np.array(
            [
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
                [6, 6, 6],
                [7, 7, 7],
                [8, 8, 8],
                [9, 9, 9],
            ]
        ),
        columns=["feature_a", "feature_b", "feature_c"],
    ),
]

output_second_val = [pd.DataFrame([[10]])]


def test_train_val_split():
    first_split = TSTrainValSplit(window_size=10, lag=5, validation_size=3)
    train, val = first_split.fit_transform(input_ts, y_ts)
    assert [
        pd.testing.assert_frame_equal(left, right)
        for left, right in zip(train, output_first_train)
    ]
    assert [
        pd.testing.assert_frame_equal(left, right)
        for left, right in zip(val, output_first_val)
    ]
    second_split = TSTrainValSplit(window_size=10, lag=10, validation_size=1)
    train, val = second_split.fit_transform(input_ts, y_ts)
    assert [
        pd.testing.assert_frame_equal(left, right)
        for left, right in zip(train, output_second_train)
    ]
    assert [
        pd.testing.assert_frame_equal(left, right)
        for left, right in zip(val, output_second_val)
    ]
