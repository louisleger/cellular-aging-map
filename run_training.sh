cd ..
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision=bf16 --num_processes=1 \
--num_machines 1 --dynamo_backend no \
-m perturbgene.main \
--pretrained_model_path 'perturbgene/model/polygene_large.json' \
--eval_data_paths '/media/rohola/ssd_storage/primary/cxg_chunk0.h5ad' \
--shard_size 10000 \
--max_length 2000 \
--num_top_genes 58604 \
--vocab_path 'perturbgene/data_utils/vocab/cxg_phenotypic_tokens_map.json' \
--included_phenotypes cell_type tissue sex development_stage disease \
--per_device_eval_batch_size 24 \
--dataloader_num_workers 4 \
--output_dir '../runs/polygene' \
mlm \
--gene_mask_prob 0.15 \
--phenotype_mask_prob 0.5 \
--train_data_paths '/media/rohola/ssd_storage/primary/cxg_chunk{1..3001}.h5ad' \
--num_hidden_layers 6 \
--num_attention_heads 6 \
--per_device_train_batch_size 32 \
--learning_rate 1e-4 \
--weight_decay 5e-2 \
--warmup_ratio 0.1 \
--num_train_epochs 3 \
--eval_steps 50000 \
--save_steps 50000

