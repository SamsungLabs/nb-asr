# coding=utf-8
import os
import pickle

from nasbench_asr.quiet_tensorflow import tensorflow as tf


class ExponentialDecay(tf.keras.callbacks.Callback):
    """
    Exponential decay-base learning rate scheduler.
    """
    def __init__(self,
                 decay_factor,
                 start_epoch=None,
                 min_lr= None,
                 verbose=0):
        """
        Args:
            decay_factor : A float value (< 1.0) indicating the factor by which the LR should be reduced
            start_epoch : At which epoch to start applying the decay, default None => start from first epoch
            min_lr : What's the lowest value for the LR allowed, default None => 0
        """

        super().__init__()
        self.decay_factor = decay_factor
        self.start_epoch = start_epoch
        if self.start_epoch is None:
            self.start_epoch = 1

        self.min_lr = min_lr
        if self.min_lr is None:
            self.min_lr = 0.0

        self.verbose = verbose
        self.epoch = 0

    def _schedule(self):

        """
        Allows the learning rate to cycle linearly within a range.
        """
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        if self.epoch < self.start_epoch:
            return tf.math.reduce_max((self.min_lr, lr))
        else:
            return tf.math.reduce_max((self.min_lr, lr * self.decay_factor))

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        self.epoch += 1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self._schedule()

        # Set the value back to the optimizer
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)

        # Adding the current LR to the logs so that it will show up on tensorboard
        logs['lr'] = float(tf.keras.backend.get_value(self.model.optimizer.lr))
