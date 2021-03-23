# Copyright (c) Facebook, Inc. and its affiliates.

from VisualBERT.mmf.common.registry import registry
from transformers.optimization import AdamW


registry.register_optimizer("adam_w")(AdamW)
