import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

from backbones import DGCNN, PointTransformer, EGNN
from utils import ExtractorMLP, MLP, CoorsNorm


class Model(nn.Module):
    def __init__(self, model_name, model_config, method_name, method_config, dataset):
        super().__init__()
        assert dataset.dataset_name in ['tau3mu', 'plbind', 'synmol'] or 'actstrack' in dataset.dataset_name
        self.dataset = dataset
        self.dataset_name = dataset.dataset_name
        self.method_name = method_name
        self.one_encoder = method_config.get('one_encoder', True)
        self.covar_dim = method_config.get('covar_dim', None)
        self.pos_coef = method_config.get('pos_coef', None)
        self.kr = method_config.get('kr', None)
        if method_name == 'psat':
            assert self.pos_coef is not None and self.kr is not None

        out_dim = 1 if dataset.num_classes == 2 else dataset.num_classes
        hidden_size = model_config['hidden_size']
        dropout_p = model_config['dropout_p']
        norm_type = model_config['norm_type']
        act_type = model_config['act_type']

        if model_name == 'dgcnn':
            Model = DGCNN
        elif model_name == 'pointtrans':
            Model = PointTransformer
        elif model_name == 'egnn':
            Model = EGNN
        else:
            raise NotImplementedError

        if model_config['pool'] == 'mean':
            self.pool = global_mean_pool
        elif model_config['pool'] == 'max':
            self.pool = global_max_pool
        elif model_config['pool'] == 'add':
            self.pool = global_add_pool
        else:
            raise NotImplementedError

        dataset.feat_info['edge_categorical_feat'], dataset.feat_info['edge_scalar_feat'] = [], dataset.pos_dim+1
        if self.dataset_name == 'plbind':
            dataset.feat_info_lig['edge_categorical_feat'], dataset.feat_info_lig['edge_scalar_feat'] = [], dataset.pos_dim+1

        raw_pos_dim = dataset.pos_dim
        if dataset.feature_type == 'only_pos':
            dataset.x_dim = 0
            dataset.x_lig_dim = 0
        elif dataset.feature_type == 'only_x':
            dataset.pos_dim = 0
            dataset.pos_lig_dim = 0
        elif dataset.feature_type == 'only_ones':
            dataset.x_dim = 0
            dataset.x_lig_dim = 0
            dataset.pos_dim = 0
            dataset.pos_lig_dim = 0
        else:
            assert dataset.feature_type == 'both_x_pos'

        aux_info = {'raw_pos_dim': raw_pos_dim, 'dataset_name': dataset.dataset_name}
        self.coors_norm = CoorsNorm()
        self.mlp_out = MLP([hidden_size, hidden_size * 2, hidden_size, out_dim], dropout_p, norm_type, act_type)
        if dataset.dataset_name != 'plbind':
            self.model = Model(dataset.x_dim, dataset.pos_dim, model_config, dataset.feat_info, aux_info=aux_info)
            self.emb_model = Model(dataset.x_dim, dataset.pos_dim, model_config, dataset.feat_info, aux_info=aux_info) if not self.one_encoder else None
        else:
            self.model = Model(dataset.x_dim, dataset.pos_dim, model_config, dataset.feat_info, dataset.n_categorical_feat_to_use, dataset.n_scalar_feat_to_use, aux_info=aux_info)
            self.emb_model = Model(dataset.x_dim, dataset.pos_dim, model_config, dataset.feat_info, aux_info=aux_info) if not self.one_encoder else None
            self.model_lig = Model(dataset.x_lig_dim, dataset.pos_lig_dim, model_config, dataset.feat_info_lig, dataset.n_categorical_feat_to_use_lig, dataset.n_scalar_feat_to_use_lig, aux_info=aux_info, is_lig=True)

        self.dim_mapping = nn.Linear(1, 8)
        self.message_weights = ExtractorMLP(8, model_config, False, out_dim=1)

    def forward(self, data, node_attn=None, edge_attn=None, node_noise=None):
        x, pos, edge_index, edge_attr = self.calc_geo_feat(data.x, data.pos, data.batch, node_noise, self.method_name)

        edge_attn = self.get_message_weights(x, pos, edge_index, data.batch) if self.method_name == 'psat' else edge_attn
        if self.dataset_name != 'plbind':
            emb = self.model(x, pos, edge_attr, edge_index, data.batch, edge_attn=edge_attn, node_attn=node_attn)
            pool_out = self.pool(emb, batch=data.batch)
        else:
            _, _, edge_index_lig, edge_attr_lig = self.calc_geo_feat(data.x_lig, data.pos_lig, data.x_lig_batch, None, self.method_name, is_lig=True)
            emb_rec = self.model(x, pos, edge_attr, edge_index, data.batch, edge_attn=edge_attn, node_attn=node_attn)
            emb_lig = self.model_lig(data.x_lig, data.pos_lig, edge_attr_lig, edge_index_lig, data.x_lig_batch)
            pool_out_rec, pool_out_lig = self.pool(emb_rec, batch=data.batch), self.pool(emb_lig, batch=data.x_lig_batch)
            pool_out = pool_out_rec + pool_out_lig
        return self.mlp_out(pool_out)

    def get_emb(self, data):
        x, pos, edge_index, edge_attr = self.calc_geo_feat(data.x, data.pos, data.batch, None, self.method_name)

        edge_attn = self.get_message_weights(x, pos, edge_index, data.batch) if self.method_name == 'psat' else None
        if self.one_encoder:
            emb = self.model(x, pos, edge_attr, edge_index, data.batch, edge_attn=edge_attn)
        else:
            emb = self.emb_model(x, pos, edge_attr, edge_index, data.batch, edge_attn=edge_attn)

        pool_out_lig = None
        if self.dataset_name == 'plbind':
            _, _, edge_index_lig, edge_attr_lig = self.calc_geo_feat(data.x_lig, data.pos_lig, data.x_lig_batch, None, self.method_name, is_lig=True)
            pool_out_lig = self.pool(self.model_lig(data.x_lig, data.pos_lig, edge_attr_lig, edge_index_lig, data.x_lig_batch), batch=data.x_lig_batch)  # pool_out_lig
        return [emb, pool_out_lig], edge_index

    def get_message_weights(self, x, pos, edge_index, batch):
        col, row = edge_index
        dist = torch.norm(pos[col] - pos[row], dim=1, p=2, keepdim=True)
        input_feat = self.dim_mapping(dist)
        return self.message_weights(input_feat, batch[col]).sigmoid()

    def calc_geo_feat(self, x, pos, batch, node_noise, method_name, is_lig=False):
        if 'actstrack' in self.dataset_name:
            pos = pos / 2955.5000 * 100 if self.pos_coef is None else pos / 2955.5000 * self.pos_coef
            pos = self.add_noise(pos, node_noise)

            pos = self.coors_norm(pos)
            edge_index = knn_graph(pos, k=5 if self.kr is None else int(self.kr), batch=batch, loop=True)
            edge_attr = self.calc_edge_attr(pos, edge_index)

        elif self.dataset_name == 'tau3mu':
            pos = pos * 1.0 if self.pos_coef is None else pos * self.pos_coef
            pos = self.add_noise(pos, node_noise)

            edge_index = radius_graph(pos, r=1.0 if self.kr is None else self.kr * self.pos_coef, loop=True, batch=batch)
            edge_attr = self.calc_edge_attr(pos, edge_index)

        elif self.dataset_name == 'synmol':
            pos = pos * 5.0 if self.pos_coef is None else pos * self.pos_coef
            pos = self.add_noise(pos, node_noise)

            edge_index = knn_graph(pos, k=5 if self.kr is None else int(self.kr), batch=batch, loop=True)
            edge_attr = self.calc_edge_attr(pos, edge_index)

        elif self.dataset_name == 'plbind':
            if is_lig:
                edge_index = radius_graph(pos, r=2.0, loop=True, batch=batch)
                edge_attr = self.calc_edge_attr(pos, edge_index)
            else:
                pos = pos * 1.0 if self.pos_coef is None else pos * self.pos_coef
                pos = self.add_noise(pos, node_noise)

                edge_index = knn_graph(pos, k=5 if self.kr is None else int(self.kr), flow='target_to_source', loop=True, batch=batch)
                edge_attr = self.calc_edge_attr(pos, edge_index)

        return x, pos, edge_index, edge_attr

    def add_noise(self, pos, node_noise):
        if node_noise is not None:
            pos[:, :self.covar_dim] = pos[:, :self.covar_dim] + node_noise
        return pos

    def calc_edge_attr(self, pos, edge_index):
        row, col = edge_index
        rel_dist = torch.norm(pos[row] - pos[col], dim=1, p=2, keepdim=True)
        coord_diff = pos[row] - pos[col]
        edge_dir = coord_diff / (rel_dist + 1e-6)
        edge_attr = torch.cat([rel_dist, edge_dir], dim=1)
        return edge_attr
