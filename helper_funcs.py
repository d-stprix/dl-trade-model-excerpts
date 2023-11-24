"""A group of functions that are used by most or all of the trading models; added in this module to reduce clutter."""

__author__ = 'Dante St. Prix'

import os
import subprocess
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision
from keras.optimizers.optimizer_v2 import optimizer_v2
import numpy as np


def set_environment_variables():
    """Loads configuration settings from a JSON file and sets environment variables.

    XLA_FLAGS and TF_GPU_THREAD_MODE are provided as strings whereas XLA_FLAGS and PATH refers to directories. Expected
    'config.json' structure:
    {
        "XLA_FLAGS": "--xla_gpu_cuda_data_dir=path_to_directory",
        "CUDA_PATH": "path_to_cuda_directory",
    }
    """
    os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2"
    os.environ['TF_GPU_THREAD_MODE'] = "gpu_private"
    with open('config.json', 'r') as file:
        config = json.load(file)
    os.environ['TF_XLA_FLAGS'] = config.get("TF_XLA_FLAGS", "")
    os.environ['PATH'] = config.get("CUDA_PATH", "") + ";" + os.environ.get('PATH', '')


def gpu_setup(num_gpus):
    """Sets up GPUs.

    Designed for NVIDIA RTX A4500 GPUs. Allocates 18 GB instead of TensorFlow's default of 17.2 GB to support training
    larger models. When running multiple GPUs, uses MultiWorkerMirroredStrategy instead of MirroredStrategy due to
    bugs in the latter resulting in poor synchronicity.

    Args:
        num_gpus: The number of GPUs to use (int).

    Returns:
        The strategy.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    virtual_device_config = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18 * 1024)]
    for i in range(num_gpus):
        tf.config.experimental.set_virtual_device_configuration(gpus[i], virtual_device_config)
    if num_gpus > 1:
        strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    else:
        strategy = None
    return strategy


def shuffle_set(sequence, daily, constant):
    """Shuffles the three portions of each dataset in place.

    The main purpose of this function is to compensate for the limited buffer size used with tf.data.Dataset.shuffle.
    Uses a seed for consistent shuffling over runs.

    Args:
        sequence: The intraday sequence array (np.ndarray).
        daily: The daily sequence array (np.ndarray).
        constant: The constant values array (np.ndarray).
    """
    np.random.seed(42)  # Set random seed for consistency across runs.
    shuffle_indices = np.random.permutation(sequence.shape[0])
    sequence[:] = sequence[shuffle_indices]
    daily[:] = daily[shuffle_indices]
    constant[:] = constant[shuffle_indices]


def create_datasets(strategy, train_sequence, train_daily, train_constant_features, train_labels, valid_sequence,
                    valid_daily, valid_constant_features, valid_labels, train_trades, valid_trades, batch_size,
                    valid_batch_size=None):
    """Configures and generates trading and validation datasets.

    After defining generators for the training and validation sets and their corresponding output signatures, the
    Numpy arrays are converted to tf.data.Dataset using tf.data.Dataset.from_generator; with the training and validation
    sets each as separate datasets. This function performs an additional round of shuffling (shuffle_set() was applied
    to the Numpy arrays previously) for each epoch to receive random instances. This function also applies batching
    and prefetching. Caching is not performed as the entire dataset currently fits in RAM.

    Args:
        strategy: The distribution strategy.
        train_sequence: The training intraday sequence array (np.ndarray).
        train_daily: The training daily sequence array (np.ndarray).
        train_constant_features: The training constant values array (np.ndarray).
        train_labels: The training labels (np.ndarray).
        valid_sequence: The validation intraday sequence array (np.ndarray).
        valid_daily: The validation daily sequence array (np.ndarray).
        valid_constant_features: The validation constant values array (np.ndarray).
        valid_labels: The validation labels (np.ndarray).
        train_trades: The number of training instances (int).
        valid_trades: The number of validation instances (int).
        batch_size: The training batch size (int).
        valid_batch_size: The validation batch size. batch_size used if unspecified (int).

    Returns:
        train_set: The training set (tf.data.Dataset).
        valid_set: The validation set (tf.data.Dataset).
    """

    def train_generator():
        for sequence, daily, constant_features, label in zip(train_sequence, train_daily, train_constant_features,
                                                             train_labels):
            yield {'intraday_input': sequence, 'daily_input': daily, 'constant_input': constant_features}, label

    def valid_generator():
        for sequence, daily, constant_features, label in zip(valid_sequence, valid_daily, valid_constant_features,
                                                             valid_labels):
            yield {'intraday_input': sequence, 'daily_input': daily, 'constant_input': constant_features}, label

    output_signature_train = (
        {'intraday_input': tf.TensorSpec(shape=train_sequence.shape[1:], dtype=train_sequence.dtype),
         'constant_input': tf.TensorSpec(shape=train_constant_features.shape[1:], dtype=train_constant_features.dtype),
         'daily_input': tf.TensorSpec(shape=train_daily.shape[1:], dtype=train_daily.dtype)},
        tf.TensorSpec(shape=train_labels.shape[1:], dtype=train_labels.dtype))

    output_signature_valid = (
        {'intraday_input': tf.TensorSpec(shape=valid_sequence.shape[1:], dtype=valid_sequence.dtype),
         'constant_input': tf.TensorSpec(shape=valid_constant_features.shape[1:], dtype=valid_constant_features.dtype),
         'daily_input': tf.TensorSpec(shape=valid_daily.shape[1:], dtype=valid_daily.dtype)},
        tf.TensorSpec(shape=valid_labels.shape[1:], dtype=valid_labels.dtype))

    mirrored_strategy = isinstance(strategy, tf.distribute.MirroredStrategy) or \
        isinstance(strategy, tf.distribute.experimental.MultiWorkerMirroredStrategy)

    if valid_batch_size is None:
        valid_batch_size = 2048
    train_set = tf.data.Dataset.from_generator(train_generator, output_signature=output_signature_train)
    valid_set = tf.data.Dataset.from_generator(valid_generator, output_signature=output_signature_valid)
    if mirrored_strategy:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_set = train_set.with_options(options)
        valid_set = valid_set.with_options(options)

    train_set = (train_set.shuffle(buffer_size=train_trades//100)
                 .batch(batch_size=batch_size, drop_remainder=True)
                 .prefetch(tf.data.AUTOTUNE))  # Preprocess.

    valid_set = (valid_set.batch(batch_size=min(valid_batch_size, valid_trades), drop_remainder=True)
                 .prefetch(tf.data.AUTOTUNE))
    if mirrored_strategy:
        train_set = strategy.experimental_distribute_dataset(train_set)
        valid_set = strategy.experimental_distribute_dataset(valid_set)
    return train_set, valid_set


def set_optimizer(optimizer):
    """Sets the model's optimizer given its string representation. Since mixed precision is used, the LossScaleOptimizer
    object is returned.

    Args:
        optimizer: The string representation of the optimizer (str).

    Returns:
        The optimizer (tf.keras.mixed_precision.LossScaleOptimizer).
    """
    if optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(rho=0.9, clipnorm=1.0)
    elif optimizer == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(clipnorm=1.0)
    elif optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
    elif optimizer == 'adamax':
        optimizer = tf.keras.optimizers.Adamax(clipnorm=1.0)
    elif optimizer == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(clipnorm=1.0)
    elif optimizer == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(clipnorm=1.0)
    elif optimizer == 'adamw':
        optimizer = tf.keras.optimizers.experimental.AdamW(clipnorm=1.0)
    elif optimizer == 'nag':
        optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True, clipnorm=1.0)
    elif optimizer == 'momentum':
        optimizer = tf.keras.optimizers.SGD(momentum=0.9, clipnorm=1.0)
    return mixed_precision.LossScaleOptimizer(optimizer)


def select_initializer(activation):
    """Selects the initializer based on the activation function.

    Based on initializers recommended in 'Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow' by Aurelion
    Geron.

    Args:
        activation: The string reprsention of the activation function (str).

    Returns:
        The string representation of the initializer (str).
    """
    if activation in ['relu', 'leaky_relu', 'elu', 'gelu', 'swish']:
        return 'he_normal'
    elif activation == 'selu':
        return 'lecun_normal'
    elif activation == 'tanh':
        return 'glorot_normal'
    else:
        raise ValueError('Activation function not recognized.')


def open_tensorboard(project_name, directory='hyperparameter_tuning'):
    """Launches TensorBoard for visualizing training logs.

    Sets the log directory to the 'tensorboard' folder found in the specified directory and creates a process
    that outputs TensorBoard to http://localhost:6006.

    Args:
        project_name: The name of the project. Unique to the model (str).
        directory: The directory where logs for all models are stored (str).
    """
    directory_base = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(directory_base, directory, project_name)
    tensorboard_logdir = 'tensorboard'
    command_to_run = f'tensorboard --logdir={tensorboard_logdir}'
    subprocess.Popen(['cmd', '/c', f'cd {directory} && {command_to_run}'])

