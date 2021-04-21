from nasbench_asr.quiet_tensorflow import tensorflow as tf


class ResetStatesCallback(tf.keras.callbacks.Callback):
    def __init__(self, trackers={}):
        super().__init__()
        self.trackers = trackers

    def on_epoch_begin(self, epoch, logs=None):
        for metric in self.trackers["train"]:
            self.trackers["train"][metric].reset_states()

    def on_test_begin(self, logs=None):
        for metric in self.trackers["test"]:
            self.trackers["test"][metric].reset_states()

