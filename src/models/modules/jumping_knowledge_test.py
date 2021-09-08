from torch import nn
from torch_geometric.nn import (
    JumpingKnowledge,
    global_max_pool, GCNConv,
)
import torch.nn.functional as F


class JK(nn.Module):

    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        self.aggregation_method = hparams["aggregation_method"]

        if hparams["num_conv_layers"] < 1:
            raise Exception("Invalid number of layers!")

        self.conv_modules = nn.ModuleList()

        self.conv_modules.append(
            GCNConv(hparams["num_node_features"], hparams["conv_size"])
        )

        for _ in range(hparams["num_conv_layers"] - 1):
            conv = GCNConv(hparams["conv_size"], hparams["conv_size"])
            self.conv_modules.append(conv)

        if self.aggregation_method == 'lstm':
            self.jk = JumpingKnowledge(self.aggregation_method,
                                       num_layers=hparams["num_conv_layers"],
                                       channels=hparams["conv_size"])
        else:
            self.jk = JumpingKnowledge(self.aggregation_method)

        if self.aggregation_method == 'cat':
            self.lin = nn.Linear(int(hparams["conv_size"] * hparams["num_conv_layers"]), hparams["lin_size"])
        else:
            self.lin = nn.Linear(int(hparams["conv_size"]), hparams["lin_size"])
        self.output = nn.Linear(hparams["lin_size"], hparams["output_size"])

    def forward(self, x, edge_index, batch, pos):
        xs = []
        for index, layer in enumerate(self.conv_modules):
            x = layer(x, edge_index)
            if index != self.hparams["num_conv_layers"]:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
            xs += [x]

        x = self.jk(xs)
        x = global_max_pool(x, batch)

        x = F.relu(self.lin(x))

        return self.output(x)

