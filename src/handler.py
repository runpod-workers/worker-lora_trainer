'''
Handler for the generation of a fine tuned lora model.
'''

import os
import shutil
import subprocess

import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import rp_download, upload_file_to_bucket

from rp_schema import INPUT_SCHEMA


def handler(job):

    job_input = job['input']

    if 'errors' in (job_input := validate(job_input, INPUT_SCHEMA)):
        return {'errors': job_input['errors']}

    # Download the zip file
    input_images = rp_download.file(job_input['zip_url'])

    if not os.path.exists('./training'):
        os.mkdir('./training')
        os.mkdir('./training/img')
        os.mkdir(
            f"./training/img/{job_input['steps']}_{job_input['instance_name']} {job_input['class_name']}")
        os.mkdir('./training/model')
        os.mkdir('./training/logs')

    # Move the images to the training folder
    shutil.move(
        input_images, f"./training/img/{job_input['steps']}_{job_input['instance_name']} {job_input['class_name']}")

    subprocess.run(f"""accelerate launch --num_cpu_threads_per_process=2 "train_network.py"
                         --enable_bucket --pretrained_model_name_or_path="/model_cache/v1-5-pruned.safetensors"
                         --train_data_dir="./training/img" --resolution=512,512 --output_dir="./training/model"
                         --logging_dir="./training/logs" --network_alpha=1 --save_model_as=safetensors --network_module=networks.lora
                         --text_encoder_lr=5e-05 --unet_lr={job_input['unet_lr']} --network_dim={job_input['network_dim']} -
                         -output_name={job['id']} --lr_scheduler_num_cycles={job_input['lr_scheduler_num_cycles']}
                         --learning_rate={job_input['learning_rate']} --lr_scheduler={job_input['lr_scheduler']}
                         --lr_warmup_steps={job_input['lr_warmup_steps']} --train_batch_size={job_input['train_batch_size']}
                         --max_train_steps={job_input['max_train_steps']} --save_every_n_epochs=0 --mixed_precision={job_input['mixed_precision']}
                         -save_precision={job_input['save_precision']} --cache_latents --optimizer_type={job_input['optimizer_type']}
                         --max_data_loader_n_workers={job_input['max_data_loader_num_workers']}
                         --bucket_reso_steps=64 --bucket_no_upscale""", shell=True, check=True)

    uploaded_lora_url = upload_file_to_bucket(
        file_name=f"{job['id']}.safetensors",
        file_location=f"./training/model/{job['id']}.safetensors",
        bucket_creds=job['s3Config'],
        bucket_name=job['s3Config']['bucketName'],
    )

    return {"lora": uploaded_lora_url}


runpod.serverless.start({"handler": handler})
