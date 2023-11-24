"""Module containing custom training loop for trading models.

The run_custom_loop function handles the entire process. Designed specifically for use with multiple GPUs. The functions
responsible for the training and validation steps are decorated with @tf.function for greater efficiency.
"""

__author__ = 'Dante St. Prix'

import functools
import math
import gc
import itertools

import tensorflow as tf
from keras.utils import io_utils

import custom_metrics
from custom_callbacks import AdjustLR, CustomEarlyStopping


def run_train_step(model, x, y_true, loss_fn):
    """Runs one training step.

    Note: Replaced by decorated version in custom loop to  avoid "ValueError: tf.function only supports singleton
    tf.Variables created on the first call that arises from applying tf.function to multiple models..."

    Args:
        x: The input features (tf.Tensor).
        y_true: The labels (tf.Tensor).
        loss_fn: The loss function (Callable).

    Returns:
        The per replica losses and metrics (Tuple[tf.Tensor, tf.Tensor]).
    """
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        per_replica_loss, per_replica_metrics = loss_fn(y_true, y_pred)
        per_replica_loss += sum(model.losses)  # Add regularization loss.
        scaled_loss = model.optimizer.get_scaled_loss(per_replica_loss)
    scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = model.optimizer.get_unscaled_gradients(scaled_grads)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return per_replica_loss, per_replica_metrics


def run_validation_step(model, x, y_true, loss_fn):
    """Runs one validation step.

    Note: Replaced by decorated version in custom loop to  avoid "ValueError: tf.function only supports singleton
    tf.Variables created on the first call that arises from applying tf.function to multiple models..."

    Args:
        x: The input features (tf.Tensor).
        y_true: The labels (tf.Tensor).
        loss_fn: The loss function (Callable).

    Returns:
        The per replica losses and metrics (Tuple[tf.Tensor, tf.Tensor]).
    """
    y_pred = model(x, training=False)
    per_replica_loss, per_replica_metrics = loss_fn(y_true, y_pred)
    per_replica_loss += sum(model.losses)  # Add regularization loss.
    return per_replica_loss, per_replica_metrics


@tf.function
def distributed_train_step(model, strategy, x, y_true, loss_fn, tf_run_train_step):
    """Coordinates one training step across multiple gpus.

    Executes the tf_run_train_step (the decorated version of run_train_step) function on each replica and then
    aggregates the results. Calculates the average loss and average metrics across all replicas.

    Args:
        model: The model trained (tf.keras.Model).
        strategy: The distribution strategy for parallelizing the training step across multiple GPUs
        (tf.distribute.Strategy).
        x: The input features (tf.Tensor).
        y_true: The labels.
        loss_fn: The loss function (Callable).
        tf_run_train_step: The function used for running one training step (Callable).

    Returns:
        The average loss and metrics (Tuple[float, Dict[str, float]]).
    """
    per_replica_losses, per_replica_metrics = strategy.run(tf_run_train_step, args=(model, x, y_true, loss_fn))
    reduced_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    average_loss = reduced_loss / strategy.num_replicas_in_sync
    average_metrics = {
        metric: strategy.reduce(tf.distribute.ReduceOp.SUM, val, axis=None) / strategy.num_replicas_in_sync
        for metric, val in per_replica_metrics.items()}
    return average_loss, average_metrics


@tf.function
def distributed_validation_loop(model, strategy, valid_set, loss_fn, tf_run_validation_step):
    """Coordinates the validation loop across multiple GPUs.

    Executes the tf_run_validation_step (the decorated version of run_validation_step) function on each replica and then
    aggregates the results. Calculates the average loss and average metrics across all replicas. Unlike with the
    distributed_training_step() function, this function iterates over the entire validation set. This is done to
    encompass the process in a single @tf.function decorator for increased efficiency. As a result, the metrics are now
    stored as individual variables as TensorFlow does not allow the modification of dict values in a decorated function.

    Args:
        model: The model trained (tf.keras.Model).
        strategy: The distribution strategy for parallelizing the training step across multiple GPUs
        (tf.distribute.Strategy).
        valid_set: The validation dataset (tf.data.Dataset).
        loss_fn: The loss function (Callable).
        tf_run_validation_step: The function used for running one validation step (Callable).

    Returns:
        The average loss and metrics (Tuple[float, Dict[str, float]]).
    """
    replicas = strategy.num_replicas_in_sync
    average_loss = 0.
    custom_loss = percent_per_trade = account_balance = trade_probability = win_rate = 0.
    num_batches = 0

    for (x, y_true) in valid_set:
        per_replica_losses, per_replica_metrics = strategy.run(tf_run_validation_step, args=(model, x, y_true, loss_fn))
        reduced_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        average_loss += reduced_loss / replicas
        # Each metric declared as variable since cannot modify dict values in tf.function.
        custom_loss += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_metrics['custom_loss'],
                                       axis=None) / replicas
        percent_per_trade += strategy.reduce(tf.distribute.ReduceOp.SUM,
                                             per_replica_metrics['percent_per_trade'], axis=None) / replicas
        account_balance += strategy.reduce(tf.distribute.ReduceOp.SUM,
                                           per_replica_metrics['account_balance'], axis=None) / replicas
        trade_probability += strategy.reduce(tf.distribute.ReduceOp.SUM,
                                             per_replica_metrics['trade_probability'], axis=None) / replicas
        win_rate += strategy.reduce(tf.distribute.ReduceOp.SUM,
                                    per_replica_metrics['win_rate'], axis=None) / replicas
        num_batches += 1

    average_metrics = {'custom_loss': custom_loss, 'percent_per_trade': percent_per_trade,
                       'account_balance': account_balance, 'trade_probability': trade_probability,
                       'win_rate': win_rate}
    num_batches = tf.cast(num_batches, dtype=tf.float32)
    return average_loss / num_batches, {metric: val / num_batches for (metric, val) in average_metrics.items()}


def step_info_generator(mini_epochs, rounds, steps_per_set):
    """Generates step information for each epoch/mini-epoch.

    This generator function yields a tuple containing two values for each mini-epoch:
    1) The number of steps in the current epoch/mini-epoch.
    2) The index of the last step in the current epoch/mini-epoch.

    The function first divides the total steps (steps_per_set) into 'mini_epochs' number of mini-epochs. It then
    divides the first 'rounds' number of epochs into mini-epochs. After completing the rounds, it continuously yields
    the total steps and the last step index for the entire set.

    Args:
        mini_epochs: The number of mini-epochs into which to divide a full epoch (int).
        rounds: The number of full epochs to divide into mini-epochs (int).
        steps_per_set: The total number of steps in a complete set (int).

    Yields:
        A tuple where the first element is the number of steps in the current epoch/mini-epoch, and the second
        element is the index of the last step in the current epoch/mini-epoch (Tuple[int, int]).
    """
    steps_per_epoch = steps_per_set // mini_epochs
    steps_cycle = itertools.cycle([steps_per_epoch for _ in range(mini_epochs-1)]
                                  + [steps_per_epoch + steps_per_set % mini_epochs])
    epoch_end_cyclical = itertools.cycle([i * steps_per_epoch-1 for i in range(1, mini_epochs)] + [steps_per_set - 1])

    for _ in range(rounds * mini_epochs):  # Iterate once per mini-epoch.
        yield next(steps_cycle), next(epoch_end_cyclical)

    while True:
        yield steps_per_set, steps_per_set - 1


def begin_epoch(epoch, step_info, epochs, stateful_metrics):
    """Obtains the epoch's step information and initializes the progress bar.

    Args:
        epoch: The current epoch number (int).
        step_info: A generator that yields tuples containing the number of steps in the current epoch and the index of
        last step (generator).
        epochs: The maximum number of epochs in the training process (int).
        stateful_metrics: A list of the number of the stateful metrics to be displayed on the progress bar (List[str])

    Returns:
        A tuple containing the number of steps in the epochs, the index of the last step in the epoch, and the
        initialized Keras progress bar object (Tuple[int, int, tf.keras.utils.Progbar]).
    """
    io_utils.print_msg(f'Epoch {epoch + 1}/{epochs}')
    steps_per_epoch, epoch_end = next(step_info)
    progbar = tf.keras.utils.Progbar(steps_per_epoch, stateful_metrics=stateful_metrics)
    return steps_per_epoch, epoch_end, progbar


def update_train_logs(loss, metric_dict, prob_ema, model, train_loss, lr, metric_records, batch_size,
                      enable_softmax, logs):
    """Updates the training logs.

    Special handling is done for the 'account_balance' metric which requires exponentiation at the end of each epoch.
    The function updates the logs dictionary in-place.

    Args:
        loss: The batch's loss (int).
        metric_dict: The batch's metrics (dict[str, float]).
        prob_ema: The EMA of the trade probability (float).
        model: The model trained (tf.keras.Model).
        train_loss: A Keras metric that tracks the training loss (tf.keras.metrics.Mean).
        lr: A Keras metric that tracks the learning rate (tf.keras.metrics.Mean).
        metric_records: A dict of Keras metric objects for tracking various training metrics
        (dict[str, tf.keras.metrics.Mean]).
        batch_size: The batch size (int).
        enable_softmax: Flag to indicate if the sofmax output is enabled. True/False for intraday/overnight models (bool).
        logs: A dictionary to store the updated logs (dict[str, float]).
    """

    if tf.math.is_finite(loss):
        train_loss.update_state(loss)
        percent_loss = -metric_dict['percent_per_trade'] * prob_ema
        logs['current loss'] = percent_loss

    percent_loss = -metric_dict['percent_per_trade'] * prob_ema
    logs['current loss'] = percent_loss

    logs['loss'] = train_loss.result()
    for metric, val in metric_dict.items():
        record = metric_records[metric]
        if tf.math.is_finite(val):
            record.update_state(val)
        if record.name != 'account_balance':
            logs[record.name] = record.result()
        else:
            logs[record.name] = custom_metrics.exponentiate_account(batch_size, record.result(), num_trades=1000)

    current_lr = tf.keras.backend.get_value(model.optimizer.lr)
    lr.update_state(current_lr)
    logs['lr'] = current_lr
    logs['sigmoid_temperature'] = model.sigmoid_temperature
    if enable_softmax:
        logs['softmax_temperature'] = model.softmax_temperature
    logs['nan_count'] = logs['nan_count'] = logs.get('nan_count', 0)
    if not tf.math.is_finite(loss):  # Place as last entry of logs.
        logs['nan_count'] += 1




def update_validation_logs(loss, metric_dict, val_loss, metric_records, val_batch_size, logs):
    """Updates the validation logs.

    Special handling is done for the 'account_balance' metric which requires exponentiation at the end of each epoch.
    The function updates the logs dictionary in-place. All logs corresponding to the validation set have a 'val_'
    prefix.

    Args:
        loss: The batch's loss (int).
        metric_dict: The batch's metrics (dict[str, float]).
        val_loss: A Keras metric that tracks the validation loss (tf.keras.metrics.Mean).
        metric_records: A dict of Keras metric objects for tracking various training metrics
        val_batch_size: The validation set batch size (int).
        logs: A dictionary to store the updated logs (dict[str, float]).
    """
    if tf.math.is_finite(loss):
        val_loss.update_state(loss)
    val_loss.update_state(loss)
    logs['val_loss'] = val_loss.result()
    for metric, val in metric_dict.items():
        record = metric_records[metric]
        if tf.math.is_finite(val):
            record.update_state(val)
        record.update_state(val)
        if record.name != 'account_balance':
            logs[f'val_{record.name}'] = record.result()
        else:
            logs[f'val_{record.name}'] = custom_metrics.exponentiate_account(val_batch_size, record.result(),
                                                              num_trades=1000)


def calculate_ema(prev_ema, val, period, overall_step):
    """Calculates the EMA of a series given the previous EMA.

    If the current step is less than the EMA period, the current step number is used as the EMA period.

    Args:
        prev_ema: The previous EMA (float).
        val: A new value (float).
        period: The EMA period (int).
        overall_step: The current step (int).
    """
    if prev_ema is None:
        return val
    elif overall_step < period:  # Take EMA with lesser period.
        alpha = 2 / (1 + overall_step)
    else:
        alpha = 2 / (1 + period)
    return (1 - alpha)*prev_ema + alpha*val


def run_custom_loop(model, strategy, callbacks, train_trades, batch_size, epochs, train_set, valid_set,
                    val_batch_size=2048, mini_epochs=4, rounds=3):
    """Runs the custom training loop.

    As the main portion of the overriden model.fit() method, this function handles the entire training process and
    returns the negated best validation percent_per_trade metric of the model. This value is negated as Keras-Tuner's
    Bayesian Optimization algorithm attempts to find models that minimize this.

    Args:
        model: The model trained (tf.keras.Model).
        strategy: The distribution strategy for parallelizing the training step across multiple GPUs
        (tf.distribute.Strategy).
        callbacks: A list containing each callback used (tf.keras.callbacks.CallbackList).
        train_trades: The number of instances in the training set (int).
        batch_size: The training batch size (int).
        epochs: The maximum number of epochs (int).
        train_set: The training dataset (tf.data.Dataset).
        valid_set: The validation dataset (tf.data.Dataset).
        val_batch_size: The validation batch size (int).
        mini_epochs: The number of mini-epochs into which a full epoch is divided (int).
        round: The number of starting epochs divided into mini-epochs (int).

    Returns:
        The best validation percent_per_trade metric (float).
    """

    # Reset functions to avoid: "ValueError: tf.function only supports singleton tf.Variables created on the first call
    # that arises from applying tf.function to multiple models..."
    tf_run_train_step = tf.function(run_train_step)
    tf_run_validation_step = tf.function(run_validation_step)
    gc.collect()  # Ensure that any prior graphs stored in GPUs are cleared.

    # Calculate mini-epoch boundaries.
    steps_per_set = train_trades // batch_size
    step_info = step_info_generator(mini_epochs, rounds, steps_per_set)
    probability_period = 1000 * 1024 // batch_size  # Probability EMA period.

    # Define loss/metrics.
    logs = {}
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    lr = tf.keras.metrics.Mean(name='lr')
    metric_names = ['custom_loss', 'percent_per_trade', 'account_balance', 'trade_probability', 'win_rate']
    metric_records = {metric_name: tf.keras.metrics.Mean(name=metric_name) for metric_name in metric_names}
    callbacks.set_model(model)

    # Loss function starts with penalty term.
    if 'intraday' in model.name:
        loss_fn = custom_metrics.intraday_loss_metric_with_penalty
        no_penalty_function = custom_metrics.intraday_loss_metric_no_penalty
        stateful_metrics = ['loss', 'sigmoid_temperature', 'softmax_temperature', 'lr', 'nan_count'] + metric_names
        enable_softmax = True
    elif 'overnight' in model.name:
        loss_fn = custom_metrics.overnight_loss_metric_with_penalty
        no_penalty_function = custom_metrics.overnight_loss_metric_no_penalty
        stateful_metrics = ['loss', 'sigmoid_temperature', 'lr', 'nan_count'] + metric_names  # No softmax output.
        enable_softmax = False
    else:
        raise ValueError(f'Name: {model.name}. Intraday or overnight should be in model.name!')

    # Reduce clutter by defining cleaner functions using functools.partial.
    partial_begin_epoch = functools.partial(begin_epoch, step_info=step_info, epochs=epochs,
                                            stateful_metrics=stateful_metrics)
    partial_update_train_logs = functools.partial(update_train_logs, model=model, train_loss=train_loss, lr=lr,
                                              metric_records=metric_records, batch_size=batch_size,
                                              enable_softmax=enable_softmax, logs=logs)
    partial_update_validation_logs = functools.partial(update_validation_logs, val_loss=val_loss,
                                              metric_records=metric_records, val_batch_size=val_batch_size, logs=logs)

    # Access AdjustLR and CustomEarlyStopping callbacks as separate variables.
    for callback in callbacks:
        if isinstance(callback, AdjustLR):
            adjust_lr = callback
        if isinstance(callback, CustomEarlyStopping):
            early_stopping = callback

    # Initialize values.
    prob_ema = None
    best_val_percent = -math.inf
    overall_step = epoch = batch = 0

    callbacks.on_train_begin(logs=logs)
    steps_per_epoch, epoch_end, progbar = partial_begin_epoch(epoch)
    callbacks.on_epoch_begin(epoch, logs=logs)

    # Training loop.
    while True:  # Loop until return conditions met.
        for real_batch, (x, y_true) in enumerate(train_set):  # Training set.
            callbacks.on_batch_begin(batch)

            # Run train step and update associated logs.
            loss, metric_dict = distributed_train_step(model, strategy, x, y_true, loss_fn, tf_run_train_step)
            if tf.math.is_finite(loss):
                prob_ema = calculate_ema(prob_ema, metric_dict['trade_probability'], probability_period, overall_step)
            partial_update_train_logs(loss, metric_dict, prob_ema)

            # Update loss function if newly profitable.
            if loss_fn is not no_penalty_function and logs['percent_per_trade'] > 0.1 and batch >= 50:
                loss_fn = no_penalty_function
                adjust_lr.record_first_profitable_step(overall_step)
                tf.print('\nDisabling loss penalty term.')

            # Handle end of training batch.
            overall_step += 1
            callbacks.on_batch_end(batch, logs=logs)
            logs.pop('current loss', None)  # Smoother loss value, should not be displayed in progbar.
            progbar.update(batch + 1, values=logs.items(), finalize=False)
            batch += 1

            if real_batch == epoch_end:  # If last batch of epoch/mini-epoch.
                logs['lr'] = lr.result()
                # Reset metric states - validation set reuses metric_records.
                for record in metric_records.values():
                    record.reset_states()

                # Run full validation set and update associated logs.
                loss, metric_dict = distributed_validation_loop(model, strategy, valid_set, no_penalty_function,
                                                                tf_run_validation_step)
                partial_update_validation_logs(loss, metric_dict)
                # Tuner gets validation percent_per_trade only if model is profitable on the training set and the
                # validation trade probability is at least 0.5%.
                if logs['percent_per_trade'] > 0 and logs['val_trade_probability'] > 0.005:
                    best_val_percent = max(best_val_percent, logs['val_percent_per_trade'].numpy())
                progbar.update(progbar.target, values=logs.items(), finalize=True)

                # Reset all states.
                logs.pop('nan_count')  # Should not be displayed in TensorBoard
                for record in [train_loss, val_loss, lr, *metric_records.values()]:
                    record.reset_states()

                # Handle end of epoch.
                callbacks.on_epoch_end(epoch, logs=logs)
                if model.stop_training or epoch == epochs - 1:  # If return conditions are met.
                    early_stopping.run_mc_model(valid_set)  # Pass values through MC model once to allow saving weights.
                    callbacks.on_train_end()
                    return -best_val_percent  # Negative since Keras-tuner tries to minimize value.

                # Begin next epoch.
                logs.clear()
                epoch += 1
                batch = 0  # Batch number resets.
                steps_per_epoch, epoch_end, progbar = partial_begin_epoch(epoch)
