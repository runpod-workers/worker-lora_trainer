INPUT_SCHEMA = {
    'zip_url': {
        'type': str,
        'required': True
    },
    'instance_name': {
        'type': str,
        'required': True
    },
    'class_name': {
        'type': str,
        'required': True
    },
    'unet_lr': {
        'type': float,
        'required': False,
        'default': 0.0001
    },
    'network_dim': {
        'type': int,
        'rqeuired': False,
        'default': 256
    },
    'lr_scheduler_num_cycles': {
        'type': int,
        'required': False,
        'default': 1
    },
    'learning_rate': {
        'type': float,
        'required': False,
        'default': 0.0001
    },
    'lr_scheduler': {
        'type': str,
        'required': False,
        'default': 'cosine'
    },
    'lr_warmup_steps': {
        'type': int,
        'required': False,
        'default': 280
    },
    'train_batch_size': {
        'type': int,
        'required': False,
        'default': 1
    },
    'max_train_steps': {
        'type': int,
        'required': False,
        'default': 1250
    },
    'mixed_precision': {
        'type': str,
        'required': False,
        'default': 'fp16'
    },
    'save_precision': {
        'type': str,
        'required': False,
        'default': 'fp16'
    },
    'optimizer_type': {
        'type': str,
        'required': False,
        'default': 'AdamW8bit'
    },
    'max_data_loader_num_workers': {
        'type': int,
        'required': False,
        'default': 0
    },
    'steps': {
        'type': int,
        'required': False,
        'default': 125
    }
}
