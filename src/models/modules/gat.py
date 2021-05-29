from torch import nn
from torch_geometric.nn import (
    GATConv,
    global_max_pool,
)


class GAT(nn.Module):

    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams

        if hparams["num_conv_layers"] < 1:
            raise Exception("Invalid number of layers!")

        activation = nn.ReLU

        self.conv_modules = nn.ModuleList()
        self.activ_modules = nn.ModuleList()

        heads = hparams.get("heads", 1)

        self.conv_modules.append(
            GATConv(hparams["num_node_features"], hparams["conv_size"], heads=heads)
        )
        self.activ_modules.append(activation())

        for _ in range(hparams["num_conv_layers"] - 1):
            conv = GATConv(heads * hparams["conv_size"], hparams["conv_size"], heads=heads)
            self.conv_modules.append(conv)
            self.activ_modules.append(activation())

        self.lin = nn.Linear(heads * hparams["conv_size"], hparams["lin_size"])

        self.output = nn.Linear(hparams["lin_size"], hparams["output_size"])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for layer, activation in zip(self.conv_modules, self.activ_modules):
            x = layer(x, edge_index)
            x = activation(x)

        x = global_max_pool(x, batch)

        x = self.lin(x)

        return self.output(x)