# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from VisualBERT.mmf.common.registry import registry
from VisualBERT.mmf.datasets.builders.vqa_hat.dataset import VQAHATDataset
from VisualBERT.mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("vqa_hat")
class VQAHATBuilder(MMFDatasetBuilder):
    def __init__(self, dataset_name="vqa_hat", dataset_class=VQAHATDataset, *args, **kwargs):
        super().__init__(dataset_name, dataset_class)
        self.dataset_class = VQAHATDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/vqa_hat/defaults.yaml"

    def load(self, *args, **kwargs):
        dataset = super().load(*args, **kwargs)
        if dataset is not None and hasattr(dataset, "try_fast_read"):
            dataset.try_fast_read()

        return dataset

    # TODO: Deprecate this method and move configuration updates directly to processors
    def update_registry_for_model(self, config):
        if hasattr(self.dataset, "text_processor"):
            registry.register(
                self.dataset_name + "_text_vocab_size",
                self.dataset.text_processor.get_vocab_size(),
            )
        if hasattr(self.dataset, "answer_processor"):
            registry.register(
                self.dataset_name + "_num_final_outputs",
                self.dataset.answer_processor.get_vocab_size(),
            )


@registry.register_builder("vqa_hat_train_val")
class VQAHATTrainValBuilder(VQAHATBuilder):
    def __init__(self, dataset_name="vqa_hat_train_val"):
        super().__init__(dataset_name)

    @classmethod
    def config_path(self):
        return "configs/datasets/vqa_hat/train_val.yaml"

@registry.register_builder("vqa_hat_test")
class VQAHATTestBuilder(VQAHATBuilder):
    def __init__(self, dataset_name="vqa_hat_test"):
        super().__init__(dataset_name)

    @classmethod
    def config_path(self):
        return "configs/datasets/vqa_hat/test.yaml"
