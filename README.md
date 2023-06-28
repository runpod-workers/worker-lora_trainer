<div align="center">

<h1>LoRA Trainer | Worker</h1>

</div>

This is a implementation https://github.com/kohya-ss/sd-scripts

1) Clone this repo
2) Cache the base model
3)

## train_network.py config

```json
{
 "model_arguments":{
    "v2": bool,
    "v_parameterization": bool,
    "pretrained_model_name_or_path": str,
    "vae": str
 },
 "additional_network_arguments":{
    "no_metadata": bool,
    "unet_lr": float,
    "text_encoder_lr": float,
    "network_weights": str,
    "network_mul": float,
    "network_args": str,
    "network_train_unet_only": bool,
    "network_train_text_encoder_only": bool,
    "training_comment": str
 },
 "dataset_arguments":{
    "debug_dataset": bool,
    "in_json": str,
    "train_data_dir": str,
    "dataset_repeats": int,
    "shuffle_caption": bool,
    "keep_tokens": int,
    "resolution": str,
    "caption_dropout_rate": int,
    "caption_tag_dropout_rate": int,
    "caption_dropout_every_n_epochs": int,
    "color_aug": bool,
    "face_crop_aug_range": int,
    "token_warmup_min": int,
    "token_warmup_min": int,
    "token_warmup_step": int
 },
 "training_arguments":{
    "output_dir": str,
    "output_name": str,
    "save_precision": str,
    "save_every_n_epochs": int,
    "save_n_epoch_ratio": int,
    "save_last_n_epochs": int,
    "save_state": bool,
    "save_last_n_epochs_state": int,
    "resume": bool,
    "train_batch_size": int,
    "max_token_length": int,
    "mem_eff_attn": bool,
    "xformers": bool,
    "max_train_epochs": int,
    "max_data_loader_n_workers": int,
    "persistent_data_loader_workers": bool,
    "seed": int,
    "gradient_checkpointing": bool,
    "gradient_accumulation_steps": int,
    "mixed_precision": str,
    "clip_skip": int,
    "logging_dir": str,
    "log_prefix": str,
    "noise_offset": float,
    "lowram": bool
 },
"sample_prompt_arguments":{
    "sample_every_n_steps": int,
    "sample_every_n_epochs": int,
    "sample_sampler": str
},
"saving_arguments":{
    "save_model_as": str
}
}
```

```bash
accelerate launch --num_cpu_threads_per_process=2 "train_network.py" \
--enable_bucket \
--pretrained_model_name_or_path="./v1-5-pruned.safetensors" \
--train_data_dir="./training/img" \
--resolution=512,512 \
--output_dir="./training/model" \
--logging_dir="./training/logs" \
--network_alpha=1 \
--save_model_as=safetensors \
--network_module=networks.lora \
--text_encoder_lr=5e-05 \
--unet_lr={UNET_LR} \
--network_dim={NETWORK_DIM} \
--output_name={SAVE_AS} \
--lr_scheduler_num_cycles={LR_SCHEDULER_NUM_CYCLES} \
--learning_rate={LEARNING_RATE} \
--lr_scheduler={LR_SCHEDULER} \
--lr_warmup_steps={LR_WARMUP_STEPS} \
--train_batch_size={TRAIN_BATCH_SIZE} \
--max_train_steps={MAX_TRAIN_STEPS} \
--save_every_n_epochs={SAVE_EVERY_N_EPOCHS} \
--mixed_precision={MIXED_PRECISION} \
--save_precision={SAVE_PRECISION} \
--cache_latents \
--optimizer_type={OPTIMIZER_TYPE} \
--max_data_loader_n_workers={MAX_DATA_LOADER_N_WORKERS} \
--bucket_reso_steps=64 \
--bucket_no_upscale
```

``````BASH
accelerate launch --num_cpu_threads_per_process 1 train_network.py \
--enable_bucket \
--pretrained_model_name_or_path="cache/v1-5-pruned.safetensors" \
--train_data_dir="/root/sd-scripts/input_imgs" \
--resolution=512,512 \
--output_dir="./training/model" \
--output_name="froggy_lora" \
--save_model_as=safetensors \
--network_module=networks.lora \
--cache_latents \
--bucket_reso_steps=64 \
--bucket_no_upscale
```
