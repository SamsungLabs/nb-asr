import torch
import torch.nn as nn

from .ops import PadConvRelu, _ops, _branch_ops


class Node(nn.Module):
    def __init__(self, filters, op_ctor, branch_op_ctors, dropout_rate=0.0):
        super().__init__()
        self.op = op_ctor(filters, filters, dropout_rate=dropout_rate)
        self.branch_ops = [ctor() for ctor in branch_op_ctors]

    def forward(self, input_list):
        assert len(input_list) == len(self.branch_ops), 'Branch op and input list have different lenghts'

        output = self.op(input_list[-1])
        edges = [output] 
        for i in range(len(self.branch_ops)):
            x = self.branch_ops[i](input_list[i])
            edges.append(x)

        return sum(edges)


class SearchCell(nn.Module): 
    def __init__(self, filters, node_configs, dropout_rate=0.0, use_norm=True):
        super().__init__()

        self.nodes = nn.ModuleList() 
        for node_config in node_configs:
            node_op_name, *node_branch_ops = node_config
            try:
                node_op_ctor = _ops[node_op_name]
            except KeyError:
                raise ValueError(f'Operation "{node_op_name}" is not implemented')

            try:
                node_branch_ctors = [_branch_ops[branch_op] for branch_op in node_branch_ops]
            except KeyError:
                raise ValueError(f'Invalid branch operations: {node_branch_ops}, expected is a vector of 0 (no skip-con.) and 1 (skip-con. present)')

            node = Node(filters=filters, op_ctor=node_op_ctor, branch_op_ctors=node_branch_ctors, dropout_rate=dropout_rate)
            self.nodes.append(node)

        self.use_norm = use_norm
        if self.use_norm:
            self.norm_layer = nn.LayerNorm(filters, eps=0.001)

    def forward(self, input):
        outputs = [input] # input is the output coming from node 0
        for node in self.nodes:
            n_out = node(outputs)
            outputs.append(n_out)
        output = outputs[-1] #last node is the output
        if self.use_norm:
            output = output.permute(0,2,1)
            output = self.norm_layer(output)
            output = output.permute(0,2,1)
        return output 


class ASRModel(nn.Module):
    def __init__(self, arch_desc, num_classes=48, use_rnn=False, use_norm=True, dropout_rate=0.0, **kwargs):
        super().__init__()

        self.arch_desc = arch_desc
        self.num_classes = num_classes
        self.use_rnn = use_rnn
        self.use_norm = use_norm
        self.dropout_rate = dropout_rate

        num_blocks = 4
        features = 80
        filters = [600, 800, 1000, 1200]
        cnn_time_reduction_kernels = [8, 8, 8, 8]
        cnn_time_reduction_strides = [1, 1, 2, 2] 
        scells_per_block = [3, 4, 5, 6]
        
        layers = nn.ModuleList()

        for i in range(num_blocks):
            layers.append(PadConvRelu(
                in_channels= features if i==0 else filters[i-1], 
                out_channels=filters[i], 
                kernel_size=cnn_time_reduction_kernels[i], 
                dilation=1,
                strides=cnn_time_reduction_strides[i],
                groups=1,
                name=f'conv_{i}'))

            # TODO: normalize axis=1
            layers.append(nn.LayerNorm(filters[i], eps=0.001))

            for j in range(scells_per_block[i]):
                cell = SearchCell(filters=filters[i], node_configs=arch_desc, use_norm=use_norm, dropout_rate=dropout_rate) 
                layers.append(cell)

        if use_rnn:
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.LSTM(input_size=filters[num_blocks-1], hidden_size=500, batch_first=True, dropout=0.0))
            layers.append(nn.Linear(in_features=500, out_features=num_classes+1))
        else:
            layers.append(nn.Linear(in_features=filters[num_blocks-1], out_features=num_classes+1))

        # self._model = nn.Sequential(*layers)
        self.model = layers

    def get_prunable_copy(self, bn=False, masks=None): 
        # bn, masks are not used in this func. 
        # Keeping them to make the code work with predictive.py
        model_new = ASRModel(arch_desc=self.arch_desc, num_classes=self.num_classes, use_rnn=self.use_rnn, use_norm=bn, dropout_rate=self.dropout_rate)
        model_new.load_state_dict(self.state_dict(), strict=False)
        model_new.train()
        return model_new

    def forward(self, input): # input is (B, F, T)
        for xx in self.model:
            if isinstance(xx, nn.LSTM):
                input = input.permute(0,2,1)
                input = xx(input)[0]
                input = input.permute(0,2,1)
            elif isinstance(xx, nn.Linear):
                input = input.permute(0,2,1)
                input = xx(input)
            elif isinstance(xx, nn.LayerNorm):
                input = input.permute(0,2,1)
                input = xx(input)
                input = input.permute(0,2,1)
            else:
                input = xx(input)
        return input

    @property
    def backend(self):
        return 'torch'
