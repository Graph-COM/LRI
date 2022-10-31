# https://github.com/vgsatorras/egnn/blob/main/qm9/models.py
# https://github.com/vgsatorras/egnn/blob/main/models/gcl.py

import torch
from torch import nn
import torch.nn.functional as F
from utils import FeatEncoder, MLP


class EGNN(nn.Module):
    def __init__(self, x_dim, pos_dim, model_config, feat_info, n_categorical_feat_to_use=-1, n_scalar_feat_to_use=-1, **kwargs):
        super().__init__()
        hidden_size = model_config['hidden_size']
        self.n_layers = model_config['n_layers']

        self.x_dim = x_dim
        self.pos_dim = pos_dim
        self.dropout_p = model_config['dropout_p']
        self.dataset_name = kwargs['aux_info']['dataset_name']
        act_fn = MLP.get_act(model_config['act_type'])()
        norm_type = model_config['norm_type']

        self.node_encoder = FeatEncoder(hidden_size, feat_info['node_categorical_feat'], feat_info['node_scalar_feat'], n_categorical_feat_to_use, n_scalar_feat_to_use)
        self.edge_encoder = FeatEncoder(hidden_size, feat_info['edge_categorical_feat'], feat_info['edge_scalar_feat'])

        edges_in_d = hidden_size if 'acts' in self.dataset_name else 0
        self.convs = nn.ModuleList()
        for _ in range(self.n_layers):
            conv = E_GCL_mask(hidden_size, hidden_size, hidden_size, edges_in_d=edges_in_d, nodes_attr_dim=0, act_fn=act_fn, norm_type=norm_type, recurrent=False, coords_weight=1.0, attention=False)
            self.convs.append(conv)

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
        edge_attr = self.edge_encoder(edge_attr) if 'acts' in self.dataset_name else None
        for i in range(self.n_layers):
            identity = x
            x, _, _ = self.convs[i](x, edge_index, pos, batch=batch, edge_attr=edge_attr, edge_attn=edge_attn)
            x = x + identity
            x = F.dropout(x, self.dropout_p, training=self.training)
        return x


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1


        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        # if self.recurrent:
        #     out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr


class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), norm_type='batch', recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        del self.coord_mlp
        self.act_fn = act_fn
        self.norm = MLP.get_norm(norm_type)(hidden_nf)

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, batch, edge_attr=None, node_attr=None, edge_attn=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        if edge_attn is not None:
            edge_feat = edge_feat * edge_attn

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        h = self.act_fn(self.norm(h))
        return h, coord, edge_attr


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
