"""
Supports training (either MLM or classification). Inference not implemented yet.
Training works across multiple GPUs on a single machine.
"""
import os
import json
import pickle

import accelerate
import torch
import torch.nn as nn
import transformers
import wandb

from perturbgene.configs import parse_args, TrainConfig
from perturbgene.data_utils import (
    DataCollatorForPhenotypicMLM, GeneTokenizer, IterableAnnDataset, collate_fn_wrapper
)
from perturbgene.eval.metrics import preprocess_logits_argmax, metrics_wrapper, set_seed
from perturbgene.model.model import Polygene
from perturbgene.data_utils.sharded_trainer import ShardedTrainer


if __name__ == "__main__":
    ##################################################
    #           General code for all tasks           #
    ##################################################
    set_seed(42)
    config = parse_args()

    distributed_state = accelerate.PartialState()

    tokenizer = GeneTokenizer(config)

    ##################################################
    #               Task-specific code               #
    ##################################################

    config: TrainConfig = config

    os.environ.update({  # https://docs.wandb.ai/guides/track/environment-variables
        "WANDB_PROJECT": "Polygene",
        "WANDB_LOG_MODEL": "false",
    })

    os.makedirs('perturbgene/' + config.output_dir, exist_ok=True) 
    working_dir = os.path.join(
        "perturbgene", f"{config.output_dir}_run_{len(os.listdir(os.path.dirname('perturbgene/' + config.output_dir)))}"
    )

    os.makedirs(working_dir, exist_ok=True) 
    with open(os.path.join(working_dir, "tokenizer.pkl"), "wb") as f: # Save tokenizer
        pickle.dump(tokenizer, f)

    # Divide `config.train_data_paths` across the processes.
    # Currently assuming that `len(train_paths)` is a perfect multiple of num_processes to make this easier.
    if len(config.train_data_paths) % distributed_state.num_processes != 0: raise NotImplementedError
    shards_per_process = len(config.train_data_paths) // distributed_state.num_processes
    rank = distributed_state.process_index
    process_train_paths = config.train_data_paths[shards_per_process * rank: shards_per_process * (rank + 1)]

    # Load the configuration for a model
    if config.pretrained_model_path.lower().endswith('json'):
        model_config = transformers.AutoConfig.from_pretrained(config.pretrained_model_path)
        model_config.num_hidden_layers = config.num_hidden_layers
        model_config.num_attention_heads = config.num_attention_heads
        model_config.max_position_embeddings = config.max_length
        model_config.vocab_size = tokenizer.vocab_size  # vocab_size and type_vocab_size determine model embeddings row length
        model_config.type_vocab_size = tokenizer.type_vocab_size
        model_config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        model_kwargs = {"attn_implementation": "flash_attention_2"} if config.use_flash_attn else dict()
        # Load relevant `model`, `data_collator`, and ` eval_metric`
        model = Polygene._from_config(model_config, **model_kwargs)
    elif "checkpoint" in config.pretrained_model_path:
        with open(config.pretrained_model_path + "../tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        tokenizer.neural_updates = {"token_values":[], "token_types": [], "token_value_str":[], "token_type_of_values":[]}
        tokenizer.flexible = True
        tokenizer.inference = False
        tokenizer.add_phenotype(config.included_phenotypes) # Add a check inside the torch model to see if these phenotypes are already added too. good enough for now if careful handling
        model = Polygene.from_pretrained(config.pretrained_model_path, attn_implementation="flash_attention_2")

    data_collator = DataCollatorForPhenotypicMLM(
        tokenizer=tokenizer,
        phenotype_mask_prob=config.phenotype_mask_prob,
        genotype_mask_prob=config.gene_mask_prob,
        )
    train_dataset = IterableAnnDataset(process_train_paths, config, tokenizer)
    eval_metric = metrics_wrapper(model, tokenizer)

    eval_dataset = IterableAnnDataset(config.eval_data_paths, config, tokenizer)

    # TODO: shuffle train, add shard_sizes arg, diff shard_size for eval

    model_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {model_total_params}")

    training_args = transformers.TrainingArguments(
        output_dir=working_dir,
        overwrite_output_dir=False,  # saving tokenizer first
        logging_steps=50,
        logging_dir=working_dir,
        logging_first_step=True,
        save_total_limit=10,
        save_strategy="steps",
        save_steps=config.save_steps,
        run_name=working_dir.split('/')[-1],
        report_to=["wandb"],
        load_best_model_at_end=True,

        warmup_ratio=config.warmup_ratio,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        
        dataloader_num_workers=config.dataloader_num_workers,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        auto_find_batch_size=config.auto_find_batch_size,
        accelerator_config={"dispatch_batches": False},
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        eval_accumulation_steps=1,
        metric_for_best_model="overall_f1",   
    )

    trainer = ShardedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=eval_metric,
        preprocess_logits_for_metrics=preprocess_logits_argmax,
    )

    # Train the model
    trainer.train()

    tokenizer.update_from_model_memory(model.config.updates_memory)
    with open(os.path.join(working_dir, "tokenizer.pkl"), "wb") as f: # Save last version of tokenizer
        pickle.dump(tokenizer, f)

    # Evaluate the model
    eval_dataset.tokenizer = tokenizer
    trainer.evaluate()

    wandb.finish()  # does Trainer not handle this?
    os._exit(0)
