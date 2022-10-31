# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dgcnn_classification.py

from torch import Tensor
from typing import Callable, Union
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor

import torch
from torch.nn import Linear
import torch.nn.functional as F

from torch_geometric.nn.inits import reset
from torch_geometric.nn.conv import MessagePassing

from utils import FeatEncoder, MLP


class DGCNN(torch.nn.Module):
    def __init__(self, x_dim, pos_dim, model_config, feat_info, n_categorical_feat_to_use=-1, n_scalar_feat_to_use=-1, **kwargs):
        super().__init__()

        hidden_size = model_config['hidden_size']
        self.n_layers = model_config['n_layers']

        self.x_dim = x_dim
        self.pos_dim = pos_dim

        self.dropout_p = model_config['dropout_p']
        norm_type = model_config['norm_type']
        act_type = model_config['act_type']

        self.node_encoder = FeatEncoder(hidden_size, feat_info['node_categorical_feat'], feat_info['node_scalar_feat'], n_categorical_feat_to_use, n_scalar_feat_to_use)
        self.edge_encoder = FeatEncoder(hidden_size, feat_info['edge_categorical_feat'], feat_info['edge_scalar_feat'])

        self.convs = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            mlp = MLP([hidden_size*3, hidden_size*2, hidden_size], 0.0, norm_type, act_type)
            self.convs.append(EdgeConv(mlp, hidden_size, norm_type, act_type, aggr='mean'))

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
            x = self.convs[i](x, edge_index, batch=batch, edge_attr=edge_attr, edge_attn=edge_attn)
            x = x + identity
            x = F.dropout(x, self.dropout_p, training=self.training)
        return x


class EdgeConv(MessagePassing):
    r"""The dynamic edge convolutional operator from the `"Dynamic Graph CNN
    for Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
    (see :class:`torch_geometric.nn.conv.EdgeConv`), where the graph is
    dynamically constructed using nearest neighbors in the feature space.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            `:obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.* defined by :class:`torch.nn.Sequential`.
        k (int): Number of nearest neighbors.
        aggr (string): The aggregation operator to use (:obj:`"add"`,
            :obj:`"mean"`, :obj:`"max"`). (default: :obj:`"max"`)
        num_workers (int): Number of workers to use for k-NN computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          batch vector :math:`(|\mathcal{V}|)` or
          :math:`((|\mathcal{V}|), (|\mathcal{V}|))`
          if bipartite *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: Callable, hidden_size, norm_type, act_type, aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, flow='source_to_target', **kwargs)

        self.nn = nn
        self.post_nn = Linear(hidden_size, hidden_size)
        self.act_fn = MLP.get_act(act_type)()
        self.norm = MLP.get_norm(norm_type)(hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor], edge_index,
            batch, edge_attr=None, edge_attn=None) -> Tensor:
        # type: (Tensor, OptTensor) -> Tensor  # noqa
        # type: (PairTensor, Optional[PairTensor]) -> Tensor  # noqa
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in DynamicEdgeConv")

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, x=x, size=None, edge_attr=edge_attr, edge_attn=edge_attn)
        out = self.post_nn(out)
        out = self.act_fn(self.norm(out))
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr, edge_attn) -> Tensor:
        msg = self.nn(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))
        if edge_attn is not None:
            return msg * edge_attn
        else:
            return msg

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn}, k={self.k})'
