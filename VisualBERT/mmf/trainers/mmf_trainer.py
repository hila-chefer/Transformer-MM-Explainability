# Copyright (c) Facebook, Inc. and its affiliates.
import logging

import omegaconf
import torch
from VisualBERT.mmf.common import typings as mmf_typings
from VisualBERT.mmf.common.dataset_loader import DatasetLoader
from VisualBERT.mmf.common.registry import registry
from VisualBERT.mmf.modules.metrics import Metrics
from VisualBERT.mmf.trainers.base_trainer import BaseTrainer
from VisualBERT.mmf.trainers.callbacks.checkpoint import CheckpointCallback
from VisualBERT.mmf.trainers.callbacks.early_stopping import EarlyStoppingCallback
from VisualBERT.mmf.trainers.callbacks.logistics import LogisticsCallback
from VisualBERT.mmf.trainers.callbacks.lr_scheduler import LRSchedulerCallback
from VisualBERT.mmf.trainers.core.callback_hook import TrainerCallbackHookMixin
from VisualBERT.mmf.trainers.core.device import TrainerDeviceMixin
from VisualBERT.mmf.trainers.core.evaluation_loop import TrainerEvaluationLoopMixin, TrainerEvaluationLoopMixinPert
from VisualBERT.mmf.trainers.core.profiling import TrainerProfilingMixin
from VisualBERT.mmf.trainers.core.reporting import TrainerReportingMixin
from VisualBERT.mmf.trainers.core.training_loop import TrainerTrainingLoopMixin
from VisualBERT.mmf.utils.build import build_model, build_optimizer
from VisualBERT.mmf.utils.general import print_model_parameters


logger = logging.getLogger(__name__)


@registry.register_trainer("mmf")
class MMFTrainer(
    TrainerCallbackHookMixin,
    TrainerTrainingLoopMixin,
    TrainerDeviceMixin,
    TrainerEvaluationLoopMixin,
    TrainerReportingMixin,
    TrainerProfilingMixin,
    BaseTrainer,
):
    def __init__(self, config: mmf_typings.DictConfig):
        super().__init__(config)

    def load(self):
        super().load()
        self.load_fp16_scaler()

        # Callbacks
        self.on_init_start()

        # Parallize model
        self.parallelize_model()

        # Callbacks
        self.on_init_end()

    def configure_callbacks(self):
        self.checkpoint_callback = CheckpointCallback(self.config, self)
        self.early_stop_callback = EarlyStoppingCallback(self.config, self)
        self.logistics_callback = LogisticsCallback(self.config, self)
        self.lr_scheduler_callback = LRSchedulerCallback(self.config, self)

        # Add callbacks for execution during events
        self.callbacks.append(self.lr_scheduler_callback)
        # checkpoint_callback needs to be called after lr_scheduler_callback so that
        # lr_scheduler_callback._scheduler.step() happens before saving checkpoints
        # (otherwise the saved last_epoch in scheduler would be wrong)
        self.callbacks.append(self.checkpoint_callback)
        self.callbacks.append(self.logistics_callback)

    def load_datasets(self):
        logger.info("Loading datasets")
        self.dataset_loader = DatasetLoader(self.config)
        self.dataset_loader.load_datasets()

        self.train_dataset = self.dataset_loader.train_dataset
        self.val_dataset = self.dataset_loader.val_dataset
        self.test_dataset = self.dataset_loader.test_dataset

        self.train_loader = self.dataset_loader.train_loader
        self.val_loader = self.dataset_loader.val_loader
        self.test_loader = self.dataset_loader.test_loader

    def load_model(self):
        logger.info("Loading model")
        attributes = self.config.model_config[self.config.model]
        # Easy way to point to config for other model
        if isinstance(attributes, str):
            attributes = self.config.model_config[attributes]

        with omegaconf.open_dict(attributes):
            attributes.model = self.config.model

        self.model = build_model(attributes)
        self.model = self.model.to(self.device)

    def load_optimizer(self):
        logger.info("Loading optimizer")
        self.optimizer = build_optimizer(self.model, self.config)

    def load_metrics(self) -> None:
        logger.info("Loading metrics")
        metrics = self.config.evaluation.get("metrics", [])
        self.metrics = Metrics(metrics)
        self.metrics_params = self.metrics.required_params

    def load_fp16_scaler(self):
        if self.training_config.fp16:
            assert (
                torch.__version__ >= "1.6"
            ), "Using fp16 requires torch version >- 1.6"
            assert self.device != torch.device("cpu"), "fp16 cannot be used on cpu"

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.training_config.fp16)

    def train(self):
        logger.info("===== Model =====")
        logger.info(self.model)
        print_model_parameters(self.model)

        if "train" not in self.run_type:
            self.inference()
            return

        self.on_train_start()
        self.training_loop()
        self.on_train_end()

        self.inference()

    def inference(self):
        dataset_type = []
        if "val" in self.run_type:
            dataset_type.append("val")
        if any(rt in self.run_type for rt in ["inference", "test", "predict"]):
            dataset_type.append("test")

        for dataset in dataset_type:
            if self.config.evaluation.predict:
                self.on_prediction_start()
                self.prediction_loop(dataset)
                self.on_prediction_end()
            else:
                self.on_test_start()
                logger.info(f"Starting inference on {dataset} set")
                report, meter = self.evaluation_loop(
                    getattr(self, f"{dataset}_loader"), use_tqdm=True
                )
                self.on_test_end(report=report, meter=meter)

@registry.register_trainer("mmf_pert")
class MMFTrainer(
    TrainerCallbackHookMixin,
    TrainerTrainingLoopMixin,
    TrainerDeviceMixin,
    TrainerEvaluationLoopMixinPert,
    TrainerReportingMixin,
    TrainerProfilingMixin,
    BaseTrainer,
):
    def __init__(self, config: mmf_typings.DictConfig):
        super().__init__(config)

    def load(self):
        super().load()
        self.load_fp16_scaler()

        # Callbacks
        self.on_init_start()

        # Parallize model
        self.parallelize_model()

        # Callbacks
        self.on_init_end()

    def configure_callbacks(self):
        self.checkpoint_callback = CheckpointCallback(self.config, self)
        self.early_stop_callback = EarlyStoppingCallback(self.config, self)
        self.logistics_callback = LogisticsCallback(self.config, self)
        self.lr_scheduler_callback = LRSchedulerCallback(self.config, self)

        # Add callbacks for execution during events
        self.callbacks.append(self.lr_scheduler_callback)
        # checkpoint_callback needs to be called after lr_scheduler_callback so that
        # lr_scheduler_callback._scheduler.step() happens before saving checkpoints
        # (otherwise the saved last_epoch in scheduler would be wrong)
        self.callbacks.append(self.checkpoint_callback)
        self.callbacks.append(self.logistics_callback)

    def load_datasets(self):
        logger.info("Loading datasets")
        self.dataset_loader = DatasetLoader(self.config)
        self.dataset_loader.load_datasets()

        self.train_dataset = self.dataset_loader.train_dataset
        self.val_dataset = self.dataset_loader.val_dataset
        self.test_dataset = self.dataset_loader.test_dataset

        self.train_loader = self.dataset_loader.train_loader
        self.val_loader = self.dataset_loader.val_loader
        self.test_loader = self.dataset_loader.test_loader

    def load_model(self):
        logger.info("Loading model")
        attributes = self.config.model_config[self.config.model]
        # Easy way to point to config for other model
        if isinstance(attributes, str):
            attributes = self.config.model_config[attributes]

        with omegaconf.open_dict(attributes):
            attributes.model = self.config.model

        self.model = build_model(attributes)
        self.model = self.model.to(self.device)

    def load_optimizer(self):
        logger.info("Loading optimizer")
        self.optimizer = build_optimizer(self.model, self.config)

    def load_metrics(self) -> None:
        logger.info("Loading metrics")
        metrics = self.config.evaluation.get("metrics", [])
        self.metrics = Metrics(metrics)
        self.metrics_params = self.metrics.required_params

    def load_fp16_scaler(self):
        if self.training_config.fp16:
            assert (
                torch.__version__ >= "1.6"
            ), "Using fp16 requires torch version >- 1.6"
            assert self.device != torch.device("cpu"), "fp16 cannot be used on cpu"

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.training_config.fp16)

    def train(self):
        logger.info("===== Model =====")
        logger.info(self.model)
        print_model_parameters(self.model)

        if "train" not in self.run_type:
            self.inference()
            return

        self.on_train_start()
        self.training_loop()
        self.on_train_end()

        self.inference()

    def inference(self):
        dataset_type = []
        if "val" in self.run_type:
            dataset_type.append("val")
        if any(rt in self.run_type for rt in ["inference", "test", "predict"]):
            dataset_type.append("test")

        for dataset in dataset_type:
            if self.config.evaluation.predict:
                self.on_prediction_start()
                self.prediction_loop(dataset)
                self.on_prediction_end()
            else:
                self.on_test_start()
                logger.info(f"Starting inference on {dataset} set")
                self.evaluation_loop(
                    getattr(self, f"{dataset}_loader"), self.on_test_end, use_tqdm=True
                )
