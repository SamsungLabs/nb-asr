# pylint: skip-file
from nasbench_asr.quiet_tensorflow import tensorflow as tf


class MeanVarianceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon, mean_initializer, variance_initializer,
                 **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.mean_initializer = mean_initializer
        self.variance_initializer = variance_initializer

    def build(self, input_shape):
        self.mean = self.add_weight(
            name="mean",
            shape=(input_shape[-1]),
            initializer=self.mean_initializer,
            trainable=False,
        )
        self.variance = self.add_weight(
            name="variance",
            shape=(input_shape[-1]),
            initializer=self.variance_initializer,
            trainable=False,
        )

    def call(self, inputs, mask=None):
        outputs = (inputs - self.mean) / tf.math.sqrt(self.variance +
                                                      self.epsilon)

        if mask is not None:
            outputs = tf.where(tf.expand_dims(mask, axis=-1), outputs, 0)

        return outputs

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon,
            "mean_initializer": self.mean_initializer,
            "variance_initializer": self.variance_initializer,
        })

        return config
