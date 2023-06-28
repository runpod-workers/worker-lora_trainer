INPUT = {
    'sample_prompts': {
        ''
    },
    # Flag
    'no_metadata': {
        'type': bool,
        'required': False,
        'default': None
    },
    'save_model_as': {
        'type': str,
        'required': False,
        'default': 'safetensors',
        'constrains': lambda saved_model: saved_model in ["ckpt", "pt", "safetensors"]
    },
    'unet_lr': {
        'type': float,
        'required': False,
        'default': None
    },
    'text_encoder_lr': {
        'type': float,
        'required': False,
        'default': None
    },
    'network_weights': {
        'type': str,
        'required': False,
        'default': None
    },
    'network_module': {
        'type': str,
        'required': False,
        'default': None
    },
    'network_dim': {
        'type': int,
        'required': False,
        'default': None
    },
    'network_alpha': {
        'type': float,
        'required': False,
        'default': 1
    },
    'network_args': {
        'type': str,
        'required': False,
        'default': None
    },
    # Flag
    'network_train_unet_only': {
        'type': bool,
        'required': False,
        'default': None
    },
    # Flag
    'network_train_text_encoder_only': {
        'type': bool,
        'required': False,
        'default': None
    },
    'training_comment': {
        'type': str,
        'required': False,
        'default': None
    }
}
