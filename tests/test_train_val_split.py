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
    np.array(
        [
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
            [14, 14, 14],
        ]
    ),
]

output_first_val = [np.array([[10]]), np.array([[15]])]

output_second_train = [
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
    np.array(
        [
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
            [10, 10, 10],
        ]
    ),
    np.array(
        [
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
        ]
    ),
    np.array(
        [
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
        ]
    ),
    np.array(
        [
            [5, 5, 5],
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
        ]
    ),
    np.array(
        [
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
            [14, 14, 14],
        ]
    ),
    np.array(
        [
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
            [14, 14, 14],
            [15, 15, 15],
        ]
    ),
    np.array(
        [
            [8, 8, 8],
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
            [14, 14, 14],
            [15, 15, 15],
            [16, 16, 16],
        ]
    ),
    np.array(
        [
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
            [14, 14, 14],
            [15, 15, 15],
            [16, 16, 16],
            [17, 17, 17],
        ]
    ),
]
output_second_val = [
    np.array([[10]]),
    np.array([[11]]),
    np.array([[12]]),
    np.array([[13]]),
    np.array([[14]]),
    np.array([[15]]),
    np.array([[16]]),
    np.array([[17]]),
    np.array([[18]]),
]

output_third_train = [
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
    )
]
output_third_val = [np.array([[10]])]


output_fourth_train = [
    np.array([]),
    np.array([]),
]
output_fourth_val = [
    np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]),
    np.array([[6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]),
]


output_fifth_train = [
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
            [10, 10, 10],
        ]
    ),
    np.array(
        [
            [6, 6, 6],
            [7, 7, 7],
            [8, 8, 8],
            [9, 9, 9],
            [10, 10, 10],
            [11, 11, 11],
            [12, 12, 12],
            [13, 13, 13],
            [14, 14, 14],
            [15, 15, 15],
        ]
    ),
]
output_fifth_val = [
    np.array([]),
    np.array([]),
]


def test_train_val_split():
    first_split = TSTrainValSplit(window_size=10, lag=5, validation_size=1)
    second_split = TSTrainValSplit(window_size=10, lag=1, validation_size=1)
    third_split = TSTrainValSplit(window_size=10, lag=10, validation_size=1)
    fourth_split = TSTrainValSplit(window_size=10, lag=5, validation_size=10)
    fifth_split = TSTrainValSplit(window_size=10, lag=5, validation_size=0)
    assert output_first_train, output_first_val == first_split.fit_transform(
        input_ts, y_ts
    )
    assert output_second_train, output_second_val == second_split.fit_transform(
        input_ts, y_ts
    )
    assert output_third_train, output_third_val == third_split.fit_transform(
        input_ts, y_ts
    )
    assert output_fourth_train, output_fourth_val == fourth_split.fit_transform(
        input_ts, y_ts
    )
    assert output_fifth_train, output_fifth_val == fifth_split.fit_transform(
        input_ts, y_ts
    )
