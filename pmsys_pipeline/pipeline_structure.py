from typing import Any, Dict, List, Union
import functools
from dataclasses import dataclass, field, fields
from abc import ABC, abstractmethod
import pandas as pd


@dataclass
class TransformerData:
    X: pd.DataFrame = field(default_factory=pd.DataFrame)
    y: pd.DataFrame = field(default_factory=pd.DataFrame)

    def apply_function(self, transformer):
        for field_ in fields(self.__class__):
            data_field = getattr(self, field_.name)
            print(transformer.function)
            if len(data_field.shape) < 2:
                data_field = data_field.values.reshape(-1, 1)
            setattr(
                    self, field_.name, pd.DataFrame(transformer.fit_transform(data_field))
                )

    def get_summary(self):
        summary_statistics = {}
        for field_ in fields(self.__class__):
            summary_statistics[field_.name] = {}
            data_field = getattr(self, field_.name)
            summary_statistics[field_.name]["mean"] = data_field.mean()
            summary_statistics[field_.name]["median"] = data_field.median()
            summary_statistics[field_.name]["max"] = data_field.max()
        return summary_statistics


class Transformer:
    def __init__(self, function):
        self.function = function
        self.arguments = {}

        functools.update_wrapper(self, function)

    def __call__(self, **arguments):
        self.arguments = arguments

    def fit_transform(self, to_transformed_data: pd.DataFrame) -> pd.DataFrame:
        return self.function(to_transformed_data, **self.arguments)


@dataclass
class PipelineModule(ABC):

    transformers: list = field(default_factory=list)
    arguments: list = field(default_factory=list)
    module_summary: dict = field(default_factory=dict)

    @classmethod
    def init_module(cls, transformers: List[Union[Transformer, callable]], arguments):
        return cls(transformers, arguments, {})

    @abstractmethod
    def get_summary(self, data: TransformerData):
        pass

    @abstractmethod
    def run(self, transformer_data) -> TransformerData:
        pass


class Pipeline:
    def __init__(self, modules: List[PipelineModule]):
        self.modules = modules

    def run(self, X, y):
        data = TransformerData(X, y)
        for module in self.modules:
            data = module.run(data)
        return data


class PreprocessingModule(PipelineModule):
    def run(self, transformer_data) -> TransformerData:
        for part, args in zip(self.transformers, self.arguments):
            part.arguments = args
            transformer_data.apply_function(part)
        self.module_summary = transformer_data.get_summary()
        return transformer_data

    def get_summary(self, transformed_data: TransformerData):
        return transformed_data.get_summary()


class ImputationModule(PipelineModule):
    def run(self, transformer_data) -> TransformerData:
        for part in self.transformers:
            transformer_data.apply_function(part)
        self.module_summary = transformer_data.get_summary()
        return transformer_data

    def get_summary(self, transformed_data: TransformerData):
        return transformed_data.get_summary()
