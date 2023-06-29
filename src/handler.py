import runpod
import boto3
import zipfile
import os
import subprocess
import shutil
import requests

from runpod.serverless.utils.validator import validate
from rp_schema import INPUT_SCHEMA

REQ_ARGS = [
    'S3_URL',
    'S3_KEY_ID',
    'S3_SECRET_KEY',
    'S3_BUCKET',
    'INSTANCE_PROMPT',
    'CLASS_PROMPT',
    'ZIP_NAME',
]


def handler(job):

    job_input = job['input']

    if 'errors' in (job_input := validate(job_input, INPUT_SCHEMA)):
        return {'errors': job_input['errors']}

    if not os.path.exists('./training'):
        os.mkdir('./training')
        os.mkdir('./training/img')
        os.mkdir(
            f"./training/img/{job_input['steps']}_{job_input['instance_name']} {job_input['class_name']}")
        os.mkdir('./training/model')
        os.mkdir('./training/logs')

    out = subprocess.run(f"""accelerate launch --num_cpu_threads_per_process=2 "train_network.py"
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

    if out.returncode == 0:
        # upload the model to the s3 bucket
        bucket.upload_file(f'./training/model/{SAVE_AS}.safetensors', f'{SAVE_AS}.safetensors')

        shutil.rmtree('./training')

        if 'ENDPOINT' in job_input:
            req = requests.post(job_input['ENDPOINT'], json={
                                "model": f'{SAVE_AS}.safetensors', "prompt": INSTANCE_PROMPT, "class": CLASS_PROMPT})

        return {
            'statusCode': 200,
        }

    return {
        'statusCode': 500,
    }


runpod.serverless.start({"handler": handler})
