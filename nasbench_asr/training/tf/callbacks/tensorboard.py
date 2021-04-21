from nasbench_asr.quiet_tensorflow import tensorflow as tf


class Tensorboard(tf.keras.callbacks.Callback):
    """
    A simple TensorBoard callback
    """
    def __init__(self, log_dir, update_freq=10):
        super().__init__()
        self.log_dir = log_dir
        self.update_freq = update_freq
        self.file_writer_train = tf.summary.create_file_writer(str(log_dir / "train"))
        self.file_writer_val = tf.summary.create_file_writer(str(log_dir / "val"))
        self.step = 0

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}

        if self.step % self.update_freq == 0:
            with self.file_writer_train.as_default():
                for k, val in logs.items():
                    tf.summary.scalar("batch/" + k, data=val, step=self.step)
        self.step += 1

    def on_epoch_end(self, epoch, logs=None):
        with self.file_writer_val.as_default():
            for k, val in logs.items():
                tf.summary.scalar("epoch/" + k, data=val, step=epoch+1)
