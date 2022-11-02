from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import pandas as pd


def get_feature_histogram_summary(
    batches: List[Union[pd.DataFrame, np.array]]
) -> Dict[str, List[Any]]:
    histogram_summary: Dict = {"feature_histograms": []}
    for batch in batches:
        batch_histograms: Dict = {}
        for i in range(0, batch.shape[1]):
            feature = pd.DataFrame(batch).iloc[:, i]
            batch_histograms[feature.name] = feature.value_counts(bins=10)
        histogram_summary["feature_histograms"].append(batch_histograms)
    return histogram_summary


def get_correlation_summary(
    batches: List[Union[pd.DataFrame, np.array]]
) -> Dict[str, List[Any]]:
    correlation_summary: Dict = {"feature_correlation": []}
    for batch in batches:
        correlation_summary["feature_correlation"].append(pd.DataFrame(batch).corr())
    return correlation_summary
