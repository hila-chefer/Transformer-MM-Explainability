# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from abc import ABC
from typing import Any, Dict, Tuple, Type

import torch
import tqdm
from VisualBERT.mmf.common.meter import Meter
from VisualBERT.mmf.common.report import Report
from VisualBERT.mmf.common.sample import to_device
from VisualBERT.mmf.utils.distributed import is_master
from VisualBERT.mmf.models.transformers.backends import ExplanationGenerator
from VisualBERT import perturbation_arguments

logger = logging.getLogger(__name__)

class TrainerEvaluationLoopMixin(ABC):
    def evaluation_loop(
        self, loader, use_tqdm: bool = False, single_batch: bool = False
    ) -> Tuple[Dict[str, Any], Type[Meter]]:
        meter = Meter()

        with torch.no_grad():
            self.model.eval()
            disable_tqdm = not use_tqdm or not is_master()
            combined_report = None

            for batch in tqdm.tqdm(loader, disable=disable_tqdm):
                report = self._forward(batch)
                self.update_meter(report, meter)

                # accumulate necessary params for metric calculation
                if combined_report is None:
                    combined_report = report
                else:
                    combined_report.accumulate_tensor_fields_and_loss(
                        report, self.metrics.required_params
                    )
                    combined_report.batch_size += report.batch_size

                if single_batch is True:
                    break

            combined_report.metrics = self.metrics(combined_report, combined_report)
            self.update_meter(combined_report, meter, eval_mode=True)

            # enable train mode again
            self.model.train()

        return combined_report, meter

    def prediction_loop(self, dataset_type: str) -> None:
        reporter = self.dataset_loader.get_test_reporter(dataset_type)
        with torch.no_grad():
            self.model.eval()
            logger.info(f"Starting {dataset_type} inference predictions")

            while reporter.next_dataset():
                dataloader = reporter.get_dataloader()

                for batch in tqdm.tqdm(dataloader):
                    prepared_batch = reporter.prepare_batch(batch)
                    prepared_batch = to_device(prepared_batch, torch.device("cuda"))
                    with torch.cuda.amp.autocast(enabled=self.training_config.fp16):
                        model_output = self.model(prepared_batch)
                    report = Report(prepared_batch, model_output)
                    reporter.add_to_report(report, self.model)

            logger.info("Finished predicting")
            self.model.train()

class TrainerEvaluationLoopMixinPert(ABC):
    def evaluation_loop(self, loader, on_test_end, use_tqdm: bool = False):
        self.model.eval()
        expl = ExplanationGenerator.SelfAttentionGenerator(self.model)

        method = perturbation_arguments.args.method
        pert_type = "pos" if perturbation_arguments.args.is_positive_pert else "neg"
        modality = "text" if perturbation_arguments.args.is_text_pert else "image"
        num_samples = perturbation_arguments.args.num_samples
        method_expl = {"transformer_attribution": expl.generate_transformer_att,
                       "ours_no_lrp": expl.generate_ours,
                       "partial_lrp": expl.generate_partial_lrp,
                       "raw_attn": expl.generate_raw_attn,
                       "attn_gradcam": expl.generate_attn_gradcam,
                       "rollout": expl.generate_rollout}

        i = 0
        # saving cams per method for all the samples
        self.model.eval()
        disable_tqdm = not use_tqdm or not is_master()
        if modality == "image":
            steps = [0, 0.5, 0.75, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
        else:
            steps = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
        step_acc = [0] * len(steps)
        print("test type {0} pert type {1} expl type {2}".format(modality, pert_type, method))
        for batch in tqdm.tqdm(loader, disable=disable_tqdm):
            method_cam = method_expl[method](batch)

            if pert_type == "pos":
                method_cam *= -1
            if modality == "image":
                input_mask = batch['input_mask']
                bbox_scores = method_cam[0, input_mask.sum(1):]
                image_boxes_len = len(bbox_scores)
                image_features = batch['image_feature_0'].clone()
                image_bboxes = batch['image_info_0']['bbox'][0].copy()
                for step_idx, step in enumerate(steps):
                    curr_num_tokens = int((1 - step) * image_boxes_len)
                    # find top step boxes
                    _, top_bboxes_indices = bbox_scores.topk(k=curr_num_tokens, dim=-1)
                    top_bboxes_indices = top_bboxes_indices.cpu().data.numpy()

                    # remove the top step boxes from the batch info
                    batch['image_feature_0'] = image_features[:, top_bboxes_indices, :]
                    batch['image_info_0']['bbox'][0] = image_bboxes[top_bboxes_indices]
                    batch['image_info_0']['max_features'] = torch.tensor(curr_num_tokens).to(batch['image_feature_0'].device).view(1)
                    batch['image_info_0']['num_boxes'][0] = curr_num_tokens

                    report = self._forward(batch)
                    step_acc[step_idx] += report["targets"][0,report["scores"].argmax()].item()

                i += 1
                if i > num_samples:
                    break
            else:
                input_mask = batch['input_mask'].clone()
                # the CLS here is ?
                cls_index = (input_mask.sum(1) - 2).item()
                seg_ids = batch["segment_ids"].clone()
                # we don't count the ? token since it's the equivalent to CLS here
                # and we want to keep the CLS intact
                text_scores = method_cam[0, 1:cls_index]
                text_len = len(text_scores)
                input_ids = batch['input_ids'].clone()
                tokens = batch['tokens'].copy()
                for step_idx, step in enumerate(steps):
                    curr_num_tokens = int((1 - step) * text_len)
                    # find top step tokens
                    _, top_bboxes_indices = text_scores.topk(k=curr_num_tokens, dim=-1)
                    top_bboxes_indices = top_bboxes_indices.cpu().data.numpy()
                    # sorting for positional embedding
                    top_bboxes_indices = list(top_bboxes_indices)
                    # add the last 2 tokens (CLS+SEP)
                    top_bboxes_indices = [0, cls_index, cls_index+1] +\
                                 [top_bboxes_indices[i] + 1 for i in range(len(top_bboxes_indices))]
                    top_bboxes_indices = sorted(top_bboxes_indices)

                    # modify the first tokens of the input mask
                    input_mask_indices = top_bboxes_indices + \
                                         [i for i in range(input_mask.sum(1), input_mask.shape[1])]

                    # remove the top step boxes from the batch info
                    batch['input_ids'] = input_ids[:, top_bboxes_indices]
                    batch['tokens'] = [[tokens[0][i] for i in top_bboxes_indices]]
                    batch['input_mask'] = input_mask[:, input_mask_indices]
                    batch["segment_ids"] = seg_ids[:, input_mask_indices]

                    report = self._forward(batch)
                    step_acc[step_idx] += report["targets"][0, report["scores"].argmax()].item()

                i += 1
                if i > num_samples:
                    break
        print("pert type {0}".format(pert_type))
        step_acc = [acc / num_samples * 100 for acc in step_acc]
        print(step_acc)
