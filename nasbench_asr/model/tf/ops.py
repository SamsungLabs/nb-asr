from nasbench_asr.quiet_tensorflow import tensorflow as tf

FUTURE_CONTEXT = 4 # 40ms look ahead

def get_activation(params_activation):
    def activation(inputs):
        return getattr(tf.keras.activations, params_activation["name"])(
            inputs, **params_activation["kwargs"])

    return activation

class PadConvRelu(tf.keras.Model):
    def __init__(self, kernel_size, dialation, filters, strides, groups=1, dropout_rate=0, name='PadConvRelu'):
        super(PadConvRelu, self).__init__(name=name)

        if int(FUTURE_CONTEXT / strides) >= (kernel_size-strides):
            rpad = kernel_size-strides
            lpad = 0
        else:
            rpad = int(FUTURE_CONTEXT / strides)
            lpad = int(kernel_size - 1 - rpad)

        padding = tf.keras.layers.ZeroPadding1D(padding=(lpad, rpad))
        conv1d = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, groups=groups, kernel_regularizer=tf.keras.regularizers.L2())
        #activation = tf.keras.layers.Activation('relu')
        activation = tf.keras.layers.Activation(get_activation({"name": "relu", "kwargs": {"max_value": 20}}))
        dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.layer = tf.keras.Sequential([padding, conv1d, activation, dropout])

    def call(self, x, training=None):
        return self.layer(x, training=training)


class Linear(tf.keras.Model):
    def __init__(self, units, dropout_rate=0, name='Linear'):
        super(Linear, self).__init__(name=name)
        dense = tf.keras.layers.Dense(units=units)
        activation = tf.keras.layers.Activation(get_activation({"name": "relu", "kwargs": {"max_value": 20}}))
        dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.layer = tf.keras.Sequential([dense, activation, dropout])

    def call(self, x, training=None):
        return self.layer(x, training=training)


class Identity(tf.keras.Model):
    def __init__(self, name='Identity'):
        super(Identity, self).__init__(name=name)

    def call(self, x):
        return x

class Zero(tf.keras.Model):
    def __init__(self, name='Zero'):
        super(Zero, self).__init__(name=name)

    def call(self, x, training=None):
        return x*0.0

DROPOUT_RATE=0.2
# OPS_old = {
#     'linear' : lambda filters: Linear(units=filters, dropout_rate=DROPOUT_RATE),
#     'conv3' : lambda filters: PadConvRelu(kernel_size=3, dialation=1, filters=filters, strides=1, groups=100, dropout_rate=DROPOUT_RATE, name='conv3'),
#     'conv3d2' : lambda filters: PadConvRelu(kernel_size=3, dialation=2, filters=filters, strides=1, groups=100, dropout_rate=DROPOUT_RATE, name='conv3d2'),
#     'conv5' : lambda filters: PadConvRelu(kernel_size=5, dialation=1, filters=filters, strides=1, groups=100, dropout_rate=DROPOUT_RATE, name='conv5'),
#     'conv5d2' : lambda filters: PadConvRelu(kernel_size=5, dialation=2, filters=filters, strides=1, groups=100, dropout_rate=DROPOUT_RATE, name='conv5d2'),
#     }

OPS = {
    'linear' : lambda filters: Linear(units=filters, dropout_rate=DROPOUT_RATE),
    'conv5' : lambda filters: PadConvRelu(kernel_size=5, dialation=1, filters=filters, strides=1, groups=100, dropout_rate=DROPOUT_RATE, name='conv3'),
    'conv5d2' : lambda filters: PadConvRelu(kernel_size=5, dialation=2, filters=filters, strides=1, groups=100, dropout_rate=DROPOUT_RATE, name='conv3d2'),
    'conv7' : lambda filters: PadConvRelu(kernel_size=7, dialation=1, filters=filters, strides=1, groups=100, dropout_rate=DROPOUT_RATE, name='conv5'),
    'conv7d2' : lambda filters: PadConvRelu(kernel_size=7, dialation=2, filters=filters, strides=1, groups=100, dropout_rate=DROPOUT_RATE, name='conv5d2'),
    'none': lambda filters:  Zero(name='none')
    }

BRANCH_OPS = {
    'none' : Zero(name='none'),
    'skip_connect' : Identity(name='skip_connect')
}

def norm_op(): 
    return tf.keras.layers.LayerNormalization()

OPS_LIST = [OPS['linear'], OPS['conv5'], OPS['conv5d2'], OPS['conv7'], OPS['conv7d2'], OPS['none']] 
BRANCH_OPS_LIST = [BRANCH_OPS['skip_connect'], BRANCH_OPS['none']]




