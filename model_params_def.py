from improvelib.utils import str2bool

preprocess_params = [
    {"name": "num_row",
     "type": int,
     "default": 50,
     "help": "Number of pixel rows in generated image.",
    },
    {"name": "num_col",
     "type": int,
     "default": 50,
     "help": "Number of pixel columns in generated image.",
    },
    {"name": "max_step",
     "type": int,
     "default": 50000, # 50000
     "help": "The maximum number of iterations to run the IGTD algorithm, if it does not converge.",
    },
    {"name": "val_step",
     "type": int,
     "default": 500, # 500
     "help": "The number of iterations for determining algorithm convergence. If the error reduction rate.",
     },
    {"name": "fea_dist_method",
     "type": str,
     "choice": ["Pearson", "Spearman", "set"],
     "default": "Euclidean",
     "help": "Method used for calculating the pairwise distances between features.",
     },
    {"name": "image_dist_method",
     "type": str,
     "choice": ["Euclidean", "Manhattan"],
     "default": "Euclidean",
     "help": "Method used for calculating the distances between pixels in image.",
     },
    {"name": "error",
     "type": str,
     "choice": ["abs", "squared"],
     "default": "abs",
     "help": "Function for evaluating the difference between feature distance ranking and pixel distance ranking.",
     }
]
train_params = [
    {'name': 'rlr_factor',
     'type': float,
     'help': 'Learning rate reduction factor'
     },
    {'name': 'rlr_min_delta',
     'type': float,
     'help': 'Learning rate reduction minimum delta'
     },
    {'name': 'rlr_cooldown',
     'type': int,
     'help': 'Learning rate reduction cooldown'
     },
    {'name': 'rlr_min_lr',
     'type': float,
     'help': 'Learning rate reduction minimum learning rate'
     },
    {'name': 'rlr_patience',
     'type': int,
     'help': 'Learning rate reduction patience'
     },
    {'name': 'es_min_delta',
     'type': float,
     'help': 'Early stop minimum delta'
     },
    {'name': 'dropout',
     'type': float,
     'default': 0.1,
     'help': 'Dropout'
     },
    {'name': 'classification_task',
     'type': str2bool,
     'default': False,
     'help': 'Is the task classification or not'
     },
    {'name': 'cnn_activation',
     'type': str,
     'default': "relu",
     'help': 'Activation function for convolution layers'
     },
    {'name': 'train_task',
     'type': str,
     'default': "",
     'help': 'Name of training task'
     },
    {"name": "canc_col_name",
     "default": "improve_sample_id",
     "type": str,
     "help": "Column name that contains the cancer sample ids."
     },
    {"name": "drug_col_name",
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name that contains the drug ids."
     },
    {"name": "verbose",
     "type": int,
     "default": 2,
     "help": "Vebosity for model."
     },
    {"name": "conv",
     "nargs": "+",
     "default": [],
     "help": "conv."
     },
    {"name": "pool",
     "nargs": "+",
     "default": [],
     "help": "pool."
     },
    {"name": "optimizer",
     "type": str,
     "default": "Adam",
     "help": "optimizer."
     },
]
infer_params = [
    {'name': 'classification_task',
     'type': str2bool,
     'default': False,
     'help': 'Is the task classification or not'
     },
    {'name': 'inference_task',
     'type': str,
     'default': "",
     'help': 'Name of inference task'
     },
    {"name": "canc_col_name",
     "default": "improve_sample_id",
     "type": str,
     "help": "Column name that contains the cancer sample ids.",
     },
    {"name": "drug_col_name",
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name that contains the drug ids.",
     }
]