# pylint: skip-file
# coding=utf-8
import os
import sys
from nasbench_asr.quiet_tensorflow import tensorflow as tf


class Ratio(tf.keras.metrics.Metric):
    def __init__(self, name="ratio", **kwargs):
        super().__init__(name=name, **kwargs)
        self.numerator = self.add_weight(name="numerator",
                                         initializer="zeros",
                                         dtype=tf.float32)
        self.denominator = self.add_weight(name="denominator",
                                           initializer="zeros",
                                           dtype=tf.float32)

    def update_state(self, numerator_denominator):
        numerator = numerator_denominator[0]
        denominator = numerator_denominator[1]
        self.numerator.assign_add(
            tf.reduce_sum(tf.cast(numerator, dtype=tf.float32)))
        self.denominator.assign_add(
            tf.reduce_sum(tf.cast(denominator, dtype=tf.float32)))

    def result(self):
        # use of / rather than tf.math.divide_no_nan is intentional

        return self.numerator / self.denominator

    # def reset_states(self):
    #     # reset across hvd works at the same time
    #     # hvd_util.apply_hvd_allreduce_np2np(0)

    #     super().reset_states()

    # def sync_across_hvd_workers(self):
    #     # for var in self.variables:
    #     #     tmp = hvd.size() * hvd_util.apply_hvd_allreduce_np2np(var.numpy())
    #     #     var.assign(tmp)
    #     pass
