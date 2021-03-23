# Copyright (c) Facebook, Inc. and its affiliates.
import argparse


class Flags:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_core_args()

    def get_parser(self):
        return self.parser

    def add_core_args(self):
        self.parser.add_argument_group("Core Arguments")
        # TODO: Add Help flag here describing MMF Configuration
        # and point to configuration documentation
        self.parser.add_argument(
            "-co",
            "--config_override",
            type=str,
            default=None,
            help="Use to override config from command line directly",
        )
        # This is needed to support torch.distributed.launch
        self.parser.add_argument(
            "--local_rank", type=int, default=None, help="Local rank of the argument"
        )
        # perturbation configuration
        self.parser.add_argument('--method', type=str,
                            default='ours_no_lrp',
                            choices=['ours_with_lrp', 'rollout', 'partial_lrp', 'transformer_att',
                                     'raw_attn', 'attn_gradcam', 'ours_with_lrp_no_normalization', 'ours_no_lrp',
                                     'ours_no_lrp_no_norm', 'ablation_no_aggregation', 'ablation_no_self_in_10'],
                            help='')
        self.parser.add_argument('--num-samples', type=int,
                            default=10000,
                            help='')
        self.parser.add_argument('--is-positive-pert', type=bool,
                            default=False,
                            help='')
        self.parser.add_argument('--is-text-pert', type=bool,
                            default=False,
                            help='')
        self.parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="Modify config options from command line",
        )



flags = Flags()
