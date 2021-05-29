from torch import nn
from torch_geometric.nn import (
    JumpingKnowledge,
    global_max_pool,
)


class JK(nn.Module):

    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams

        if hparams["num_conv_layers"] < 1:
            raise Exception("Invalid number of layers!")

        activation = nn.ReLU

        self.conv_modules = nn.ModuleList()
        self.activ_modules = nn.ModuleList()

        self.aggregation_method = hparams["aggregation_method"]

        self.conv_modules.append(
            JumpingKnowledge(self.aggregation_method, hparams["num_node_features"], hparams["conv_size"])
        )
        self.activ_modules.append(activation())

        for _ in range(hparams["num_conv_layers"] - 1):
            conv = JumpingKnowledge(self.aggregation_method, hparams["conv_size"], hparams["conv_size"])
            self.conv_modules.append(conv)
            self.activ_modules.append(activation())

        self.lin = nn.Linear(hparams["conv_size"], hparams["lin_size"])

        self.output = nn.Linear(hparams["lin_size"], hparams["output_size"])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for layer, activation in zip(self.conv_modules, self.activ_modules):
            x = layer(x, edge_index)
            x = activation(x)

        x = global_max_pool(x, batch)

        x = self.lin(x)

        return self.output(x)