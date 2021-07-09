from typing import Optional, Union
from torch_geometric.typing import Adj, OptPairTensor, Size, PairTensor
from torch import Tensor, scatter
from torch.nn import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
import torch.nn.functional as F


class VectorSAGE(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, spatial_dim: int,
                 aggregator: str, root_weight: bool = True, **kwargs):
        kwargs.setdefault('aggr', None)
        super(VectorSAGE, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.aggregator = aggregator

        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.lin_l = Linear(in_channels[0], out_channels, bias=True)
        self.lin_pos = Linear(spatial_dim, in_channels[0], bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                pos: Union[Tensor, PairTensor],
                size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, size=size, pos=pos)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor) -> Tensor:
        pos = pos_j - pos_i
        spatial = self.lin_pos(pos)
        n_edges = spatial.size(0)
        result = spatial.reshape(n_edges, self.in_channels, -1) * x_j.unsqueeze(-1)
        return result.view(n_edges, self.in_channels)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        if self.aggregator == 'mean':
            out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
        elif self.aggregator == 'min':
            out = scatter(inputs, index, 0, None, dim_size, reduce='min')
        elif self.aggregator == 'max':
            out = scatter(inputs, index, 0, None, dim_size, reduce='max')
        else:
            raise ValueError(f'Unknown aggregator "{self.aggregator}".')
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}')
