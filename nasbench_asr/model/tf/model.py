import random
import itertools as itr

from nasbench_asr.quiet_tensorflow import tensorflow as tf

from .ops import OPS_LIST, BRANCH_OPS_LIST, norm_op, PadConvRelu
from .mean_variance_normalization import MeanVarianceNormalization


class Node(tf.keras.Model):
    def __init__(self, filters, op_idx, branch_op_idx_list):
        super().__init__()
        self._op = OPS_LIST[op_idx](filters)
        self.branch_ops = [BRANCH_OPS_LIST[i] for i in branch_op_idx_list]

    def call(self, input_list, training=None):
        assert len(input_list) == len(self.branch_ops), 'Branch op and input list have different lenghts'

        output = self._op(input_list[-1], training=training)
        edges = [output]
        for i in range(len(self.branch_ops)):
            x = self.branch_ops[i](input_list[i])
            edges.append(x)

        return tf.math.add_n(edges)


class SearchCell(tf.keras.Model): 
    def __init__(self, filters, config, num_nodes=3):
        super().__init__()

        self._nodes = list() 
        for n_config in config:
            node = Node(filters=filters, op_idx=n_config[0], branch_op_idx_list=n_config[1:])
            self._nodes.append(node)  
            
        self.norm_layer = norm_op()

    def call(self, input, training=None):
        outputs = [input] # input is the output coming from node 0
        for node in self._nodes:
            n_out = node(outputs, training=training)
            outputs.append(n_out)

        output = self.norm_layer(outputs[-1]) #use layer norm at the end of a search cell
        return output 


class ASRModel(tf.keras.Model):
    def __init__(self, arch_desc, num_classes=48, use_rnn=False, use_norm=True, dropout_rate=0.0, input_shape=None, data_norm=None, epsilon=0.001):
        super().__init__()

        self.arch_desc = list(arch_desc)
        self.num_classes = num_classes
        self.use_rnn = use_rnn
        self.use_norm = use_norm
        self.dropout_rate = dropout_rate

        cnn_time_reduction_kernels = [8, 8, 8, 8]
        cnn_time_reduction_strides = [1, 1, 2, 2]
        filters = [600, 800, 1000, 1200]
        scells_per_block = [3, 4, 5, 6]

        zipped_params = zip(cnn_time_reduction_kernels,
            cnn_time_reduction_strides,
            filters,
            scells_per_block)

        layers = []

        if input_shape is not None:
            layers.append(tf.keras.layers.Masking(input_shape=input_shape))
        else:
            layers.append(tf.keras.layers.Masking())

        if data_norm is not None:
            mean, variance = data_norm
            layers.append(MeanVarianceNormalization(epsilon, tf.keras.initializers.Constant(mean), tf.keras.initializers.Constant(variance)))

        for i, (kernel, stride, filters, cells) in enumerate(zipped_params):
            layers.append(PadConvRelu(kernel_size=kernel, strides=stride, filters=filters, dialation=1, name=f'conv_{i}'))
            layers.append(norm_op())

            for j in range(cells):
                layers.append(SearchCell(filters=filters, config=arch_desc))

        if use_rnn:
            layers.append(tf.keras.layers.LSTM(units=500, dropout=self.dropout_rate, time_major=False, return_sequences=True))

        layers.append(tf.keras.layers.Dense(self.num_classes+1))

        self._model = tf.keras.Sequential(layers)

    def call(self, input, training=None):
        return self._model(input, training=training)

