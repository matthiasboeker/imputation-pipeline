from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import List
from typing import Union

import numpy as np
import pandas as pd

from pmsys_pipeline.summary_functions import get_correlation_summary
from pmsys_pipeline.summary_functions import get_feature_histogram_summary


def parse_to_numpy_tensor(data_obj: Union[np.array, pd.DataFrame]):
    if isinstance(data_obj, pd.DataFrame):
        data_obj = data_obj.to_records(index=False)
    return np.expand_dims(data_obj, axis=len(data_obj.shape))


@dataclass(frozen=True)
class TransformerData:
    X: List[pd.DataFrame] = field(default_factory=List[pd.DataFrame])
    y: List[pd.DataFrame] = field(default_factory=List[pd.DataFrame])

    def get_summary(self):
        summary_statistics = {}
        summary_statistics.update(get_correlation_summary(self.X))
        summary_statistics.update(get_feature_histogram_summary(self.X))
        return summary_statistics


class Transformer:
    def __init__(self, function, arguments):
        self.function = function
        self.arguments = arguments

    def __call__(self, **arguments):
        self.arguments = arguments

    def fit_transform(self, to_transformed_data: pd.DataFrame) -> pd.DataFrame:
        return self.function(to_transformed_data, **self.arguments)


@dataclass
class PipelineModule(ABC):

    transformers: list = field(default_factory=list)
    module_summary: dict = field(default_factory=dict)

    @classmethod
    def init_module(cls, transformers: List[Union[Transformer, Callable]]):
        return cls(transformers, {})

    @abstractmethod
    def get_summary(self, data: TransformerData):
        pass

    @abstractmethod
    def run(self, transformer_data: TransformerData) -> TransformerData:
        pass


class Pipeline:
    def __init__(self, modules: List[PipelineModule]):
        self.modules = modules

    def run(self, X, y):
        data = TransformerData([X], [y])
        for module in self.modules:
            data = module.run(data)
        return data


def run_module(input_data: TransformerData, transformers: List[Transformer]):
    X = input_data.X
    y = input_data.y
    for transformer in transformers:
        X = [transformer.fit_transform(batch) for batch in X]
        y = [transformer.fit_transform(batch) for batch in y]
    return TransformerData(X, y)


class TSTrainTestSplittingModule(PipelineModule):
    @classmethod
    def init_module(cls, splitter):
        return cls([splitter], {})

    def run(self, transformer_data) -> TransformerData:
        (splitter,) = self.transformers
        (X,) = transformer_data.X
        (y,) = transformer_data.y
        train_batches, val_batches = splitter.fit_transform(X, y)
        output_transformer_data = TransformerData(train_batches, val_batches)
        self.module_summary = output_transformer_data.get_summary()
        return output_transformer_data

    def get_summary(self, transformed_data: TransformerData):
        return transformed_data.get_summary()


class PreprocessingModule(PipelineModule):
    def run(self, transformer_data) -> TransformerData:
        output_transformer_data = run_module(transformer_data, self.transformers)
        self.module_summary = output_transformer_data.get_summary()
        return output_transformer_data

    def get_summary(self, transformed_data: TransformerData):
        return transformed_data.get_summary()


class ImputationModule(PipelineModule):
    def run(self, transformer_data) -> TransformerData:
        output_transformer_data = run_module(transformer_data, self.transformers)
        self.module_summary = output_transformer_data.get_summary()
        return output_transformer_data

    def get_summary(self, transformed_data: TransformerData):
        return transformed_data.get_summary()
