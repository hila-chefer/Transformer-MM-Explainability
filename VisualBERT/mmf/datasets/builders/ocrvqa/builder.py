# Copyright (c) Facebook, Inc. and its affiliates.
from VisualBERT.mmf.common.registry import Registry
from VisualBERT.mmf.datasets.builders.ocrvqa.dataset import OCRVQADataset
from VisualBERT.mmf.datasets.builders.textvqa.builder import TextVQABuilder


@Registry.register_builder("ocrvqa")
class OCRVQABuilder(TextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "ocrvqa"
        self.set_dataset_class(OCRVQADataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/ocrvqa/defaults.yaml"
