<div align="center">

<h1>LoRA Trainer | Worker</h1>

</div>

This is a implementation https://github.com/kohya-ss/sd-scripts
Derived from e1pupper's RunPod worker implementation https://github.com/e1pupper/loratrainer

1) Clone this repo
2) Cache the base model
3)


## Inputs

| Name                        | Type  | Description                                             | Default     | Required |
|-----------------------------|-------|---------------------------------------------------------|-------------|:--------:|
| zip_url                     | str   | URL to the zip file containing the data                 | None        |    ✔️     |
| instance_name               | str   | Name of the model instance                              | None        |    ✔️     |
| class_name                  | str   | Name of the model class                                 | None        |    ✔️     |
| unet_lr                     | float | Learning rate of the U-Net model                        | 0.0001      |          |
| network_dim                 | int   | Dimension of the neural network                         | 256         |          |
| lr_scheduler_num_cycles     | int   | Number of cycles of the learning rate scheduler         | 1           |          |
| learning_rate               | float | Global learning rate                                    | 0.0001      |          |
| lr_scheduler                | str   | Type of the learning rate scheduler                     | 'cosine'    |          |
| lr_warmup_steps             | int   | Number of steps for the learning rate warmup            | 280         |          |
| train_batch_size            | int   | Batch size for training                                 | 1           |          |
| max_train_steps             | int   | Maximum number of training steps                        | 1250        |          |
| mixed_precision             | str   | Precision type used for mixed precision training        | 'fp16'      |          |
| save_precision              | str   | Precision type used when saving the model               | 'fp16'      |          |
| optimizer_type              | str   | Type of the optimizer used in training                  | 'AdamW8bit' |          |
| max_data_loader_num_workers | int   | Maximum number of workers for the data loader           | 0           |          |
| steps                       | int   | Number of steps to be taken during the training process | 125         |          |


### Example

```json
{
    "input":{
        "zip_url": "https://github.com/runpod-workers/sample-inputs/raw/main/images/froggy.zip",
        "instance_name": "daiton",
        "class_name": "frog",
        "unet_lr": 0.0001,
        "network_dim": 256,
        "lr_scheduler_num_cycles": 1,
        "learning_rate": 0.0001,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 280,
        "train_batch_size": 1,
        "max_train_steps": 1250,
        "mixed_precision": "fp16",
        "save_precision": "fp16",
        "optimizer_type": "AdamW8bit",
        "max_data_loader_num_workers": 0,
        "steps": 125
    }
}
```

```BASH
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
