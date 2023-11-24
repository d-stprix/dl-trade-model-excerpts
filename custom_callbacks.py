"""Contains custom callbacks used by the training models.

Includes callbacks for early stopping, learning rate scheduling and plotting metrics.
"""

__author__ = 'Dante St. Prix'

import os
import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

global TRIAL_NUM  # Keeps track of the trial number used in naming the model save files.
TRIAL_NUM = 0

class CustomEarlyStopping(tf.keras.callbacks.Callback):
    """Customized class for stopping training when the model stops improving.

    After start_epoch, if the monitored value does not improve by loss_factor within a set number of epochs determined
    by the patience, the trial is terminated. An assumption is made that improvement corresponds to a decrease in the
    monitored value. Moreover, if the model stops trading (< 0.5% trade probability) or fails to become profitable
    within the first 8 epochs, training is also stopped. Additionally, saves the most profitable model weights (highest
    val_percent_per_trade) if there exists an epoch during which the model is profitable on both the training and
    validation sets. Both the original and Monte Carlo versions of the model are saved.

    Clarifications:
    - The first epoch where termination can occur is start_epoch + patience.
    - If loss is the monitored value, the value must be below: loss*(1-loss_factor) within patience epochs.

    Args:
        directory: The location where model weights are saved.
        mc_model: The Monte Carlo version of the model.
        monitor: Quantity to be monitored (generally the loss).
        loss_factor: The proportion by which the monitored value must decrease to avoid early stopping.
        start_epoch: The epoch after which improvement is checked.
        patience: The number of epochs during which the loss must drop to the threshold amount.
    """
    def __init__(self, directory, mc_model, monitor='loss', loss_factor=0.01, start_epoch=5, patience=5):
        super().__init__()
        self.directory = directory
        self.mc_model = mc_model
        self.monitor = monitor
        self.start_epoch = start_epoch
        self.patience = patience
        self.loss_factor = loss_factor
        self.wait = 0
        self.best = None
        self.req_loss = None
        self.prev_loss = None
        self.best_valid = 0.
        self.best_weights = None
        self.stop_current_epoch = False
        self.trial_num = TRIAL_NUM
        self.state = 0

    def on_train_begin(self, logs=None):
        """Resets attributes to default values."""
        global TRIAL_NUM
        TRIAL_NUM += 1
        self.trial_num = TRIAL_NUM
        self.best_valid = 0.
        self.wait = 0
        self.best = None
        self.req_loss = None
        self.prev_loss = None
        self.best_weights = None
        self.stop_current_epoch = False
        self.state = 0

    def on_epoch_end(self, epoch, logs=None):
        """Checks if stopping conditions met."""
        global TRIAL_NUM
        current_loss = logs[self.monitor]
        profitable = True if logs['percent_per_trade'] > 0 and logs['val_percent_per_trade'] > 0 else False

        if profitable and (self.best_weights is None or logs['val_percent_per_trade'] > self.best_valid) and \
                logs['val_trade_probability'] > 0.005:
            # Training set must be profitable to update self.best_weights.
            tf.print(f'\nBest val_percent_per_trade so far ({logs["val_percent_per_trade"]:.4f}%), updating '
                     f'self.best_weights.')
            self.best_weights = self.model.get_weights()
            self.best_valid = logs['val_percent_per_trade']

        if epoch >= 8 and logs['percent_per_trade'] < 0:
            tf.print('\nFailed to become profitable for training set within first 8 epochs, terminating.')
            self.model.stop_training = True
        elif logs['trade_probability'] < 0.005:
            tf.print('\nModel stopped trading, terminating')
            self.model.stop_training = True

        elif self.state == 0:  # Do nothing until start_epoch.
            if epoch == self.start_epoch:
                self.best = current_loss
                loss_multiplier = (1 - self.loss_factor) if self.best >= 0 else (1 + self.loss_factor)
                self.req_loss = self.best * loss_multiplier
                self.state = 1
        elif self.state == 1:
            large_improvement = current_loss < self.req_loss
            if large_improvement:  # Reset counter
                self.best = current_loss
                loss_multiplier = (1 - self.loss_factor) if self.best >= 0 else (1 + self.loss_factor)
                self.req_loss = self.best * loss_multiplier
                self.wait = 0
            else:
                self.wait += 1
                if self.wait == self.patience:
                    tf.print('\nInsufficient progress made, terminating.')
                    self.model.stop_training = True
        self.prev_loss = current_loss

    def run_mc_model(self, valid_set):
        # To avoid error: cannot be saved either because the input shape is not available or because the forward pass of
        # the model is not defined.To define a forward pass, please override `Model.call()`. To specify an input shape,
        # either call `build(input_shape)` directly
        self.mc_model.predict(valid_set, steps=1, verbose=0)

    def on_train_end(self, epoch, logs=None):
        """Saves the most profitable model weights to the given directory."""
        if self.best_weights is not None:
            tf.print('Saving best model weights to disk.')
            self.model.set_weights(self.best_weights)
            self.mc_model.set_weights(self.best_weights)
            model_path = self.directory + f'\\{self.best_valid:.2}_percent_model_{self.trial_num}'
            mc_model_path = self.directory + f'\\mc_{self.best_valid:.2}_percent_model_{self.trial_num}'
            self.model.save(model_path, save_format='tf', overwrite=True)
            self.mc_model.save( mc_model_path, save_format='tf', overwrite=True)
        else:
            tf.print('No epochs were profitable. Not saving.')


class AdjustLR(tf.keras.callbacks.Callback):
    """Learning rate scheduler.

    Starts with an initial learning rate of 2e-4 (0.2 for Adadelta). Once the loss function is toggled, this learning
    rate remains for an additional 20,000 steps (for batch size of 1024, 40,000 for 512). After this point, this
    callback decreases the learning rate while oscillating it between half and double its center value.

    Args:
        batch_size: The size of each batch.
        halving_rate: The number of steps during which the learning rate halves.
        sin_period: Learning rate's oscillation period in units of steps.
    """
    def __init__(self, model, batch_size, halving_rate=5000, sin_period=500):
        super().__init__()
        self.batch_ratio = 1024 // batch_size
        self.p1 = 100 * self.batch_ratio
        self.p2 = 500 * self.batch_ratio
        self.lr_decay = 2 ** -(1 / (halving_rate * self.batch_ratio))
        self.sin_period = sin_period
        self.state = 0
        self.overall_step = 0
        self.first_profitable_step = None
        self.model = model

        # Multiply learning rate by 1000 if Adadelta optimizer.
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        if isinstance(self.model.optimizer.inner_optimizer, tf.keras.optimizers.Adadelta):
            tf.keras.backend.set_value(self.model.optimizer.lr, lr * 1000)

    def record_first_profitable_step(self, overall_step):
        """Called when the loss function is toggled. Records the step."""
        self.first_profitable_step = overall_step

    def on_batch_end(self, batch, logs=None):
        """Adjusts the learning rate according to the strategy mentioned in the docstring."""
        if self.state == 0:  # Highest LR until profitable.
            if self.first_profitable_step:
                lr = tf.keras.backend.get_value(self.model.optimizer.lr)
                new_lr = lr
                self.lr_0 = new_lr / 2
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                self.state = 1
        elif self.state == 1:  # Lower LR and maintain.
            if self.overall_step == self.first_profitable_step + 20000 * self.batch_ratio:
                self.start_step = self.overall_step
                self.state = 2
        elif self.state == 2:  # Oscillate LR while decreasing.
            self.lr_0 *= self.lr_decay
            sinusoid_term = 2**math.cos(2*math.pi*(self.overall_step-self.start_step)/(self.sin_period * self.batch_ratio))
            new_lr = self.lr_0 * sinusoid_term
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        self.overall_step += 1


class MetricPlotter(tf.keras.callbacks.Callback):
    """Displays the specified metric using an interactive Matplotlib graph.

    Plots take the form x = step number, y = metric value. If EMA periods are specified, the corresponding EMAs are
    calculated and added to the plots as solid lines; while the individual points are shown are cyan points. If no EMAs
    are specified, then the points are plotted as solid lines instead.

    Args:
        metric: The metric to plot. Must be included in the logs dict.
        ema_len: A sequence containing EMA periods to include in the plot.
        window_length: The length of the plot window (displays the last window_length points). If set to None, all steps
        are plotted.
        frequency: The number of steps between each plot update. Used to ensure insignificant impact on training time.
        semilog: Whether the plot should be a semilog plot (used for plotting the learning rate).
        name: The name of the metric shown in the plot title and legend.
    """
    def __init__(self, metric, ema_len=(), window_length=None, frequency=20, semilog=False, name=None):
        super().__init__()
        self.x = []
        self.y = []
        self.overall_step = 0
        self.metric = metric
        if name is not None:
            self.name = name
        else:
            self.name = metric
        self.ema = [[] for _ in ema_len]
        self.ema_period = ema_len
        self.alpha = [2 / (1 + period) for period in self.ema_period]
        self.window_length = window_length
        self.semilog = semilog
        self.frequency = frequency

    def calculate_ema(self, val):
        """Calculates the EMA of the metric."""
        for (ema, period, alpha) in zip(self.ema, self.ema_period, self.alpha):
            if len(ema) == 0:
                if np.isnan(val):
                    ema.append(0.)
                else:
                    ema.append(val)
            elif np.isnan(val):
                ema.append(ema[-1])
            elif self.overall_step < period:  # Average of all values.
                ema.append(np.nansum(self.y) / len(self.y))
            else:
                ema.append((1 - alpha) * ema[-1] + alpha * val)

    def on_train_begin(self, logs=None):
        """Creates the plot at the start of training."""
        plt.ion()  # turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(9, 6))

    def on_batch_end(self, batch, logs=None):
        """Calculates the EMAs and updates the plot."""
        if len(self.x) == 0:
            self.x.append(0)
        else:
            self.x.append(self.x[-1] + 1)
        val = logs.get(self.metric, np.nan)
        self.y.append(val)
        self.calculate_ema(val)

        if self.window_length is not None and len(self.x) > self.window_length:
            self.x.pop(0)
            self.y.pop(0)
            for ema in self.ema:
                ema.pop(0)

        if self.overall_step != 0 and self.overall_step % self.frequency == 0:
            self.ax.clear()
            # Update the plot.
            plot = self.ax.semilogy if self.semilog else self.ax.plot
            if len(self.ema) == 0:
                plot(self.x, self.y, label=self.name, color='c')
            else:
                plot(self.x, self.y, label=self.name, color='c', marker='.', linestyle='none', alpha=0.15)
            for (ema, period) in zip(self.ema, self.ema_period):
                self.ax.plot(self.x, ema, label=f'{period} EMA')

            if self.name != 'Learning Rate':
                # Set y-bounds.
                lower_bound = np.nanpercentile(self.y, 1)
                upper_bound = max(np.nanpercentile(self.y, 99), 0.1)
                self.ax.set_ylim(lower_bound, upper_bound)

            self.ax.set_xlabel('Step')
            self.ax.set_ylabel(self.name)
            self.ax.set_title(f'{self.name} over Steps')
            if len(self.ema) != 0:
                self.ax.legend()
            self.ax.grid(which="major", linewidth=1)
            self.ax.grid(which="minor", linewidth=0.2)
            self.ax.minorticks_on()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        self.overall_step += 1

    def on_train_end(self, logs=None):
        """Closes the figure at the end of training."""
        plt.close(self.fig)
