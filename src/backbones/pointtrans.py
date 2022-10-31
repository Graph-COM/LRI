# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/point_transformer_classification.py

from torch import Tensor
from typing import Callable, Optional, Tuple, Union
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin

from torch_geometric.nn.inits import reset
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

from utils import FeatEncoder


class PointTransformer(torch.nn.Module):
    def __init__(self, x_dim, pos_dim, model_config, feat_info, n_categorical_feat_to_use=-1, n_scalar_feat_to_use=-1, **kwargs):
        super().__init__()

        hidden_size = model_config['hidden_size']
        self.n_layers = model_config['n_layers']

        self.x_dim = x_dim
        self.pos_dim = pos_dim
        self.dropout_p = model_config['dropout_p']
        self.raw_pos_dim = kwargs['aux_info']['raw_pos_dim']

        self.node_encoder = FeatEncoder(hidden_size, feat_info['node_categorical_feat'], feat_info['node_scalar_feat'], n_categorical_feat_to_use, n_scalar_feat_to_use)
        self.edge_encoder = FeatEncoder(hidden_size, feat_info['edge_categorical_feat'], feat_info['edge_scalar_feat'])

        self.convs = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.convs.append(TransformerBlock(hidden_size, hidden_size, pos_dim=self.raw_pos_dim))

    def forward(self, x, pos, edge_attr, edge_index, batch, edge_attn=None, node_attn=None):
        if self.x_dim == 0 and self.pos_dim != 0:
            feats = pos
        elif self.x_dim != 0 and self.pos_dim == 0:
            feats = x
        elif self.x_dim == 0 and self.pos_dim == 0:
            feats = torch.ones(x.shape[0], 1, device=x.device)
        else:
            feats = torch.cat([x, pos], dim=1)

        x = self.node_encoder(feats)
        edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            identity = x
            x = self.convs[i](x, pos, edge_index, edge_attr=edge_attr, edge_attn=edge_attn)
            x = x + identity
            x = F.dropout(x, self.dropout_p, training=self.training)
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, pos_dim):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)
        self.transformer = PointTransformerConv(in_channels, out_channels, pos_dim=pos_dim)

    def forward(self, x, pos, edge_index, edge_attr=None, edge_attn=None):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index, edge_attr=edge_attr, edge_attn=edge_attn)
        x = self.lin_out(x).relu()
        return x


class PointTransformerConv(MessagePassing):
    r"""The Point Transformer layer from the `"Point Transformer"
    <https://arxiv.org/abs/2012.09164>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i =  \sum_{j \in
        \mathcal{N}(i) \cup \{ i \}} \alpha_{i,j} \left(\mathbf{W}_3
        \mathbf{x}_j + \delta_{ij} \right),

    where the attention coefficients :math:`\alpha_{i,j}` and
    positional embedding :math:`\delta_{ij}` are computed as

    .. math::
        \alpha_{i,j}= \textrm{softmax} \left( \gamma_\mathbf{\Theta}
        (\mathbf{W}_1 \mathbf{x}_i - \mathbf{W}_2 \mathbf{x}_j +
        \delta_{i,j}) \right)

    and

    .. math::
        \delta_{i,j}= h_{\mathbf{\Theta}}(\mathbf{p}_i - \mathbf{p}_j),

    with :math:`\gamma_\mathbf{\Theta}` and :math:`h_\mathbf{\Theta}`
    denoting neural networks, *i.e.* MLPs, and
    :math:`\mathbf{P} \in \mathbb{R}^{N \times D}` defines the position of
    each point.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        pos_nn : (torch.nn.Module, optional): A neural network
            :math:`h_\mathbf{\Theta}` which maps relative spatial coordinates
            :obj:`pos_j - pos_i` of shape :obj:`[-1, 3]` to shape
            :obj:`[-1, out_channels]`.
            Will default to a :class:`torch.nn.Linear` transformation if not
            further specified. (default: :obj:`None`)
        attn_nn : (torch.nn.Module, optional): A neural network
            :math:`\gamma_\mathbf{\Theta}` which maps transformed
            node features of shape :obj:`[-1, out_channels]`
            to shape :obj:`[-1, out_channels]`. (default: :obj:`None`)
        add_self_loops (bool, optional) : If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          positions :math:`(|\mathcal{V}|, 3)` or
          :math:`((|\mathcal{V_s}|, 3), (|\mathcal{V_t}|, 3))` if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, pos_nn: Optional[Callable] = None,
                 attn_nn: Optional[Callable] = None,
                 add_self_loops: bool = False, pos_dim=3, **kwargs):
        kwargs.setdefault('aggr', 'add')  # https://github.com/pyg-team/pytorch_geometric/pull/5332
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.pos_nn = pos_nn
        if self.pos_nn is None:
            self.pos_nn = Linear(pos_dim, out_channels)

        self.attn_nn = attn_nn
        self.lin = Linear(in_channels[0], out_channels, bias=False)
        self.lin_src = Linear(in_channels[0], out_channels, bias=False)
        self.lin_dst = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.pos_nn)
        if self.attn_nn is not None:
            reset(self.attn_nn)
        self.lin.reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj, edge_attr=None, edge_attn=None
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            alpha = (self.lin_src(x), self.lin_dst(x))
            x: PairTensor = (self.lin(x), x)
        else:
            alpha = (self.lin_src(x[0]), self.lin_dst(x[1]))
            x = (self.lin(x[0]), x[1])

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairTensor, pos: PairTensor, alpha: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, alpha=alpha, size=None, edge_attr=edge_attr, edge_attn=edge_attn)
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                alpha_i: Tensor, alpha_j: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int], edge_attr=None, edge_attn=None) -> Tensor:

        delta = self.pos_nn(pos_i - pos_j)
        alpha = alpha_i - alpha_j + delta
        if self.attn_nn is not None:
            alpha = self.attn_nn(alpha)
        alpha = softmax(alpha, index, ptr, size_i)

        if edge_attr is not None:
            msg = alpha * (x_j + delta + edge_attr)
        else:
            msg = alpha * (x_j + delta)

        if edge_attn is not None:
            msg = msg * edge_attn
        return msg

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
