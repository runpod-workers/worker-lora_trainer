import runpod
import boto3
import zipfile
import os
import subprocess
import shutil
import requests

REQ_ARGS = [
    'S3_URL',
    'S3_KEY_ID',
    'S3_SECRET_KEY',
    'S3_BUCKET',
    'INSTANCE_PROMPT',
    'CLASS_PROMPT',
    'ZIP_NAME',
]


def handler(event):

    for arg in REQ_ARGS:
        if arg not in event['input']:
            return {
                'statusCode': 400,
                'body': f'{arg} not found in event'
            }

    s3 = boto3.resource('s3',
                        endpoint_url=event['input']['S3_URL'],
                        aws_access_key_id=event['input']['S3_KEY_ID'],
                        aws_secret_access_key=event['input']['S3_SECRET_KEY']
                        )

    bucket = s3.Bucket(event['input']['S3_BUCKET'])

    ZIP_FILE = event['input']['ZIP_NAME']
    INSTANCE_PROMPT = event['input']['INSTANCE_PROMPT']
    CLASS_PROMPT = event['input']['CLASS_PROMPT']
    SAVE_AS = event['input']['SAVE_AS'] if 'SAVE_AS' in event['input'] else INSTANCE_PROMPT
    UNET_LR = event['input']['UNET_LR'] if 'UNET_LR' in event['input'] else 0.0001
    NETWORK_DIM = event['input']['NETWORK_DIM'] if 'NETWORK_DIM' in event['input'] else 256
    LR_SCHEDULER_NUM_CYCLES = event['input']['LR_SCHEDULER_NUM_CYCLES'] if 'LR_SCHEDULER_NUM_CYCLES' in event['input'] else 1
    LEARNING_RATE = event['input']['LEARNING_RATE'] if 'LEARNING_RATE' in event['input'] else 0.0001
    LR_SCHEDULER = event['input']['LR_SCHEDULER'] if 'LR_SCHEDULER' in event['input'] else 'cosine'
    LR_WARMUP_STEPS = event['input']['LR_WARMUP_STEPS'] if 'LR_WARMUP_STEPS' in event['input'] else 280
    TRAIN_BATCH_SIZE = event['input']['TRAIN_BATCH_SIZE'] if 'TRAIN_BATCH_SIZE' in event['input'] else 1
    MAX_TRAIN_STEPS = event['input']['MAX_TRAIN_STEPS'] if 'MAX_TRAIN_STEPS' in event['input'] else 1250
    SAVE_EVERY_N_EPOCHS = 0
    MIXED_PRECISION = event['input']['MIXED_PRECISION'] if 'MIXED_PRECISION' in event['input'] else 'fp16'
    SAVE_PRECISION = event['input']['SAVE_PRECISION'] if 'SAVE_PRECISION' in event['input'] else 'fp16'
    OPTIMIZER_TYPE = event['input']['OPTIMIZER_TYPE'] if 'OPTIMIZER_TYPE' in event['input'] else 'AdamW8bit'
    MAX_DATA_LOADER_N_WORKERS = event['input']['MAX_DATA_LOADER_N_WORKERS'] if 'MAX_DATA_LOADER_N_WORKERS' in event['input'] else 0

    if 'STEPS' not in event['input']:
        STEPS = 125
    else:
        STEPS = event['input']['STEPS']

    if not os.path.exists('./training'):
        os.mkdir('./training')
        os.mkdir('./training/img')
        os.mkdir(f'./training/img/{STEPS}_{INSTANCE_PROMPT} {CLASS_PROMPT}')
        os.mkdir('./training/model')
        os.mkdir('./training/logs')

    bucket.download_file(
        f'{ZIP_FILE}.zip', f'./training/img/{STEPS}_{INSTANCE_PROMPT} {CLASS_PROMPT}/{ZIP_FILE}.zip')

    with zipfile.ZipFile(f'./training/img/{STEPS}_{INSTANCE_PROMPT} {CLASS_PROMPT}/{ZIP_FILE}.zip', 'r') as zip_ref:
        zip_ref.extractall(f'./training/img/{STEPS}_{INSTANCE_PROMPT} {CLASS_PROMPT}')

    os.remove(f'./training/img/{STEPS}_{INSTANCE_PROMPT} {CLASS_PROMPT}/{ZIP_FILE}.zip')

    out = subprocess.run(
        f'accelerate launch --num_cpu_threads_per_process=2 "train_network.py" --enable_bucket --pretrained_model_name_or_path="/model_cache/v1-5-pruned.safetensors" --train_data_dir="./training/img" --resolution=512,512 --output_dir="./training/model" --logging_dir="./training/logs" --network_alpha=1 --save_model_as=safetensors --network_module=networks.lora --text_encoder_lr=5e-05 --unet_lr={UNET_LR} --network_dim={NETWORK_DIM} --output_name={SAVE_AS} --lr_scheduler_num_cycles={LR_SCHEDULER_NUM_CYCLES} --learning_rate={LEARNING_RATE} --lr_scheduler={LR_SCHEDULER} --lr_warmup_steps={LR_WARMUP_STEPS} --train_batch_size={TRAIN_BATCH_SIZE} --max_train_steps={MAX_TRAIN_STEPS} --save_every_n_epochs={SAVE_EVERY_N_EPOCHS} --mixed_precision={MIXED_PRECISION} --save_precision={SAVE_PRECISION} --cache_latents --optimizer_type={OPTIMIZER_TYPE} --max_data_loader_n_workers={MAX_DATA_LOADER_N_WORKERS} --bucket_reso_steps=64 --bucket_no_upscale', shell=True)

    if out.returncode == 0:
        # upload the model to the s3 bucket
        bucket.upload_file(f'./training/model/{SAVE_AS}.safetensors', f'{SAVE_AS}.safetensors')

        shutil.rmtree('./training')

        if 'ENDPOINT' in event['input']:
            req = requests.post(event['input']['ENDPOINT'], json={
                                "model": f'{SAVE_AS}.safetensors', "prompt": INSTANCE_PROMPT, "class": CLASS_PROMPT})

        return {
            'statusCode': 200,
        }

    return {
        'statusCode': 500,
    }


runpod.serverless.start({"handler": handler})
