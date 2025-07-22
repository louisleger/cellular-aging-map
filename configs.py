"""
Parses user arguments into the relevant Config class.
The Config classes are unnecessary but help for PyCharm type hints.
This comes with the cost that any update to the arguments requires changing both the parser and a Config class.
"""

import argparse
import json
import os
from dataclasses import dataclass
from braceexpand import braceexpand


@dataclass
class BaseConfig:
    subcommand: str
    bin_edges: list[float]
    pretrained_model_path: str
    shard_size: int
    eval_data_paths: list[str]
    max_length: int
    num_top_genes: int

    vocab_path: str
    included_phenotypes: list[str]

    use_flash_attn: bool

    output_dir: str
    per_device_eval_batch_size: int
    dataloader_num_workers: int
    auto_find_batch_size: bool

    def __post_init__(self):
        """ Postprocessing and basic sanity checks """

        if self.bin_edges is not None:
            assert min(self.bin_edges) >= 0, "Assuming no expressions will fall strictly below the lowest bin_edge."

        assert os.path.isfile(self.vocab_path), f"{self.vocab_path=} is not a file."

        # could also try opening to check it's a JSON file
        assert self.vocab_path.lower().endswith(".json"), f"Expected json path but got {self.vocab_path=}"
        if self.included_phenotypes is None: self.included_phenotypes = []

        assert self.max_length <= self.num_top_genes + 2 + len(self.included_phenotypes) # ???? max is lower bound??
        self.eval_data_paths = [path for paths in self.eval_data_paths for path in braceexpand(paths)]


@dataclass
class TrainConfig(BaseConfig):
    train_data_paths: list[str]
    num_hidden_layers: int
    num_attention_heads: int

    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    save_steps: int
    eval_steps: int

    gene_mask_prob: float
    phenotype_mask_prob: float

    def __post_init__(self):
        super().__post_init__()
        eval_paths = set(self.eval_data_paths)
        self.train_data_paths = [path for paths in self.train_data_paths for path in braceexpand(paths) if path not in eval_paths]
        assert eval_paths & set(self.train_data_paths) == set(), "same data in train and eval"

        


def parse_args(args: list[str] = None) -> BaseConfig:
    """
    Get command line arguments.

    Optional arguments to `parse_args`:
        args: If specified, will parse these `args`. Otherwise, defaults to `sys.argv`.
    """
    parser = argparse.ArgumentParser(description="Arguments for training/evaluating a model. Any arguments without "
                                                 "descriptions correspond directly to transformer.TrainingArguments.")
    subparsers = parser.add_subparsers(help="Sub-command help", dest="subcommand")

    mlm_subparser = subparsers.add_parser("mlm", description="Training with an MLM objective.")

    parser.add_argument("--bin_edges", nargs="+", type=float,
                        help="Provided `n` edges, will partition the gene expression values into `n + 1` bins, these edges are shared across all genes.",
                        default=[0.5, 1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--pretrained_model_path", type=str, help="Path to a pretrained model to initialize from.")
    parser.add_argument("--shard_size", default=10000, type=int, help="The number of observations in each AnnData shard.")
    parser.add_argument("--eval_data_paths", nargs="+", type=str, help="One or more (possibly `braceexpand`-able) paths to validation h5ad file(s).")
    parser.add_argument("--max_length", type=int, required=True, help="The maximum sequence length for the transformer.")
    parser.add_argument("--num_top_genes", type=int, required=False, default=int(6e5), help="The number of highest variable genes to use.")

    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the JSON file mapping token_types to tokens.")
    parser.add_argument("--included_phenotypes", nargs="*", type=str, help="The phenotypes to include in the model input.")

    parser.add_argument("--use_flash_attn", action="store_false", help="Whether to use Flash Attention 2. If true, also expects `fp16`.")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_eval_batch_size", type=int, required=True)
    parser.add_argument("--dataloader_num_workers", default=4, type=int)
    parser.add_argument("--auto_find_batch_size", action="store_true")

    mlm_subparser.add_argument("--train_data_paths", nargs="+", type=str, help="One or more (possible `braceexpand`-able) paths to training h5ad file(s).")
    mlm_subparser.add_argument("--num_hidden_layers", type=int, required=True, help="Number of hidden layers to use in the model.")
    mlm_subparser.add_argument("--num_attention_heads", type=int, required=True, help="Number of attention heads to use in the model.")

    mlm_subparser.add_argument("--num_train_epochs", type=int, required=True)
    mlm_subparser.add_argument("--per_device_train_batch_size", type=int, required=True)
    mlm_subparser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    mlm_subparser.add_argument("--learning_rate", type=float, required=True)
    mlm_subparser.add_argument("--weight_decay", type=float, required=True)
    mlm_subparser.add_argument("--warmup_ratio", default=0.0, type=float)
    mlm_subparser.add_argument("--save_steps", type=int, required=True)
    mlm_subparser.add_argument("--eval_steps", type=int, required=True)

    mlm_subparser.add_argument("--gene_mask_prob", type=float, required=True, help="The probability a gene token will be masked.")
    mlm_subparser.add_argument("--phenotype_mask_prob", type=float, required=True, help="The probability a phenotype token will be masked.")

    args = parser.parse_args(args)
    if args.subcommand is None:
        return BaseConfig(**vars(args))

    return TrainConfig(**vars(args))
