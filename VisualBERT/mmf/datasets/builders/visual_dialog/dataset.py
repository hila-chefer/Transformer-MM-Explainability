import copy
import json

import torch
from VisualBERT.mmf.common.sample import Sample
from VisualBERT.mmf.datasets.builders.visual_dialog.database import VisualDialogDatabase
from VisualBERT.mmf.datasets.builders.vqa2 import VQA2Dataset


class VisualDialogDataset(VQA2Dataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            config,
            dataset_type,
            imdb_file_index,
            dataset_name="visual_dialog",
            *args,
            **kwargs
        )

        discriminative = config.discriminative
        self._discriminative = discriminative.enabled
        self._return_indices = discriminative.return_indices
        self._no_unk = config.no_unk
        self._return_history = config.return_history
