"""Performs hyperparameter tuning.

Allows the user to select one of the six architectures: intraday/overnight RNN/wavenet/transformer. Calls all necessary
modules and functions to load dataset into RAM, set up the GPUs, creates two copies of each model (original and Monte
Carlo versions), sets up TensorBoard, trains each model and selects new hyperparameter combinations to maximize
generalizability.
"""

__author__ = 'Dante St. Prix'

import os
import shutil
from pathlib import Path
import warnings

import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras import backend as K
from tensorflow.keras import mixed_precision

import intraday_data
import overnight_data
import hypermodels
import helper_funcs
import patch_utils
from custom_loop import run_custom_loop
from custom_callbacks import CustomEarlyStopping
from custom_callbacks import AdjustLR
from custom_callbacks import MetricPlotter

K.clear_session()
K.set_epsilon(1e-4)
helper_funcs.set_environment_variables()
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.optimizer.set_jit(True)
policy16 = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy16)

warnings.filterwarnings('ignore', category=UserWarning, message='.*Converting sparse IndexedSlices.*')  # Reduce clutter.

MAX_EPOCHS = 60
SAMPLE = 0  # samples: 0 = full dataset, 1 = first 100k instances, 2 = first 10k instances.
NEW_EXPERIMENT = True  # Have option to delete files from previous trials.in
INTRADAY_LENGTH = 128  # Look-back period.
DAILY_LENGTH = 128  # For daily candles.
NUM_GPUS = 2
START_LR = 2e-4  # Automatically scales values for Adagrad and Adadelta.
PROFILE = True
DEBUG = False

#  Select one of:
#  ['RNN Intraday', 'RNN Overnight', 'Transformer Intraday', 'Transformer Overnight', 'Wavenet Intraday',
#  'Wavenet Overnight']
MODEL_TYPE = 'Transformer Intraday'


def choose_hypermodel():
    """Selects and returns a specified hypermodel class."""
    match MODEL_TYPE:
        case 'RNN Intraday':
            return hypermodels.RNNIntraday
        case 'RNN Overnight':
            return hypermodels.RNNOvernight
        case 'Transformer Intraday':
            return hypermodels.TransformerIntraday
        case 'Transformer Overnight':
            return hypermodels.TransformerOvernight
        case 'Wavenet Intraday':
            return hypermodels.WavenetIntraday
        case 'Wavenet Overnight':
            return hypermodels.WavenetOvernight
        case _:
            raise ValueError('Model type not recognized!')


def main():

    hypermodel = choose_hypermodel()
    project_name = MODEL_TYPE.replace(' ', '_').lower()
    directory = 'hyperparameter_tuning'

    # Double check if user wants to delete data from prior trials.
    if NEW_EXPERIMENT and os.path.exists(f'hyperparameter_tuning\\{project_name}') and input('Enter Y to delete old '
                                                                                             'tuning data: ') == 'Y':
        print('Deleting data from prior trials.')
        shutil.rmtree(f'hyperparameter_tuning\\{project_name}')

    # Load dataset.
    if 'intraday' in project_name:
        data, num_targets = intraday_data.get_dataset(SAMPLE)
    else:
        data = overnight_data.get_dataset(SAMPLE)
    train_sequence = data['train_sequence']
    train_daily = data['train_daily']
    train_constant = data['train_constant']
    valid_sequence = data['valid_sequence']
    valid_daily = data['valid_daily']
    valid_constant = data['valid_constant']

    # Set distribution strategy.
    strategy = helper_funcs.gpu_setup(NUM_GPUS)
    patch_utils.apply_patches(strategy)  # Patch TensorFlow functions (for handling NaN losses).

    # Process dataset.
    num_trades = train_sequence.shape[0]
    print(f'{num_trades/1e6:.2f} M trades in dataset.')
    num_symbols = max(int(np.amax(train_constant[:, 0]) + 1), int(np.amax(valid_constant[:, 0]) + 1))
    num_hour_min = 390
    num_sequence_features = train_sequence.shape[2]
    num_daily_features = train_daily.shape[2]
    print('Shuffling training set...')
    helper_funcs.shuffle_set(train_sequence, train_daily, train_constant)
    print('Shuffling validation set...')
    helper_funcs.shuffle_set(valid_sequence, valid_daily, valid_constant)  # Ordering affects loss function.
    train_trades, valid_trades = train_sequence.shape[0], valid_sequence.shape[0]
    if 'intraday' in project_name:
        train_labels = train_constant[:, -num_targets:]
        train_constant_features = train_constant[:, :-num_targets]
        valid_labels = valid_constant[:, -num_targets:]
        valid_constant_features = valid_constant[:, :-num_targets]
    else:
        train_labels = train_constant[:, -1]
        train_constant_features = train_constant[:, :-1]
        valid_labels = valid_constant[:, -1]
        valid_constant_features = valid_constant[:, :-1]

    # Define model.
    class Hypermodel(kt.HyperModel):
        def build(self, hp):
            """Builds model to be used by keras-tuner."""
            if 'intraday' in project_name:
                model_params = (hp, num_symbols, num_hour_min, num_targets, num_sequence_features, num_daily_features)
            else:
                model_params = (hp, num_symbols, num_hour_min, num_sequence_features, num_daily_features)

            with strategy.scope():
                model = hypermodel(*model_params)  # RNN, Wavenet or transformer.
                model.build(input_shape={'intraday_input': (None, *train_sequence.shape[1:]),
                                         'daily_input': (None, *train_daily.shape[1:]),
                                         'constant_input': (None, *train_constant_features.shape[1:])})
            model.summary()
            tf.keras.backend.set_value(model.optimizer.lr, START_LR)
            return model

        def build_mc_model(self, hp):
            """Builds model to be used by keras-tuner."""
            if 'intraday' in project_name:
                model_params = (hp, num_symbols, num_hour_min, num_targets, num_sequence_features, num_daily_features)
            else:
                model_params = (hp, num_symbols, num_hour_min, num_sequence_features, num_daily_features)

            with strategy.scope():
                mc_model = hypermodel(*model_params, mc=True)
                mc_model.build(input_shape={'intraday_input': (None, *train_sequence.shape[1:]),
                                         'daily_input': (None, *train_daily.shape[1:]),
                                         'constant_input': (None, *train_constant_features.shape[1:])})
            return mc_model

        def fit(self, hp, model, *args, **kwargs):
            K.clear_session()
            mc_model = self.build_mc_model(hp)

            epochs = MAX_EPOCHS
            batch_size = hp.get('batch_size')
            batch_ratio = 1024 // batch_size
            model_directory = os.path.join(os.getcwd(), directory, project_name)
            early_stopping = CustomEarlyStopping(model_directory, mc_model=mc_model, monitor='custom_loss',
                                                 loss_factor=0.01, start_epoch=14, patience=2)
            adjust_lr = AdjustLR(model, batch_size)
            lr_plotter = MetricPlotter('lr', frequency=500*batch_ratio, semilog=True, name='Learning Rate')
            loss_plotter = MetricPlotter('current loss', ema_len=(100*batch_ratio, 500*batch_ratio),
                                         frequency=500*batch_ratio, name='Percent Loss')
            callbacks = kwargs['callbacks'] + [early_stopping, adjust_lr, lr_plotter, loss_plotter]
            callbacks = tf.keras.callbacks.CallbackList(callbacks)
            with strategy.scope():
                train_set, valid_set = helper_funcs.create_datasets(strategy, train_sequence, train_daily,
                                                                    train_constant_features, train_labels,
                                                                    valid_sequence, valid_daily,
                                                                    valid_constant_features, valid_labels, train_trades,
                                                                    valid_trades, batch_size)
            return run_custom_loop(model, strategy, callbacks, train_trades, batch_size, epochs, train_set, valid_set,
                                   mini_epochs=4, rounds=3)

    # Define tuner.
    tuner = kt.BayesianOptimization(
        hypermodel=Hypermodel(),
        distribution_strategy=strategy,
        max_trials=100000, directory=directory, project_name=project_name)

    # Tensorboard.
    root_logdir = Path(tuner.project_dir) / "tensorboard"
    helper_funcs.open_tensorboard(project_name, directory)
    if PROFILE:
        tensorboard_cb = tf.keras.callbacks.TensorBoard(root_logdir, histogram_freq=1, profile_batch=(600, 610))
    else:
        tensorboard_cb = tf.keras.callbacks.TensorBoard(root_logdir, histogram_freq=1)

    if DEBUG:
        tf.debugging.enable_check_numerics()

    # Perform hyperparameter search.
    tuner.search(callbacks=[tensorboard_cb])


if __name__.endswith('__main__'):
    main()
