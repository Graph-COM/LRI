# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/gnn_explainer.html
# https://github.com/RexYing/gnn-model-explainer/blob/master/explainer/explain.py

import torch
import torch.nn as nn
from math import sqrt


class BernMask(nn.Module):

    def __init__(self, clf, extractor, criterion, config):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device

        self.pred_loss_coef = config['pred_loss_coef']
        self.size_loss_coef = config['size_loss_coef']
        self.mask_ent_loss_coef = config['mask_ent_loss_coef']
        self.iter_per_sample = config['iter_per_sample']
        self.iter_lr = config['iter_lr']

    def __loss__(self, mask, clf_logits, clf_labels, epoch, warmup):
        pred_loss = self.criterion(clf_logits, clf_labels.float())
        if warmup:
            return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}

        size_loss = mask.mean()
        mask_ent_reg = -mask * (mask + 1e-10).log() - (1 - mask) * (1 - mask + 1e-10).log()
        mask_ent_loss = mask_ent_reg.mean()

        pred_loss = self.pred_loss_coef * pred_loss
        size_loss = self.size_loss_coef * size_loss
        mask_ent_loss = self.mask_ent_loss_coef * mask_ent_loss

        loss = pred_loss + size_loss + mask_ent_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'size': size_loss.item(), 'ent': mask_ent_loss.item()}
        return loss, loss_dict

    def _initialize_masks(self, x, init="normal"):
        N = x.size()[0]
        std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N))
        self.node_mask = torch.nn.Parameter(torch.FloatTensor(N, 1).normal_(1, std))

    def _clear_masks(self):
        self.node_mask = None

    def forward_pass(self, data, epoch, warmup, **kwargs):
        if warmup:
            clf_logits = self.clf(data)
            loss, loss_dict = self.__loss__(None, clf_logits, data.y, epoch, warmup)
            return loss, loss_dict, clf_logits, None, None, None, None

        self._clear_masks()
        _, edge_index = self.clf.get_emb(data)
        with torch.no_grad():
            original_clf_logits = self.clf(data)

        self._initialize_masks(data.x)
        self.to(data.x.device)

        parameters = [self.node_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.iter_lr)

        for _ in range(self.iter_per_sample):
            optimizer.zero_grad()
            node_mask = torch.sigmoid(self.node_mask)
            edge_mask = self.node_attn_to_edge_attn(node_mask, edge_index)
            masked_clf_logits = self.clf(data, edge_attn=edge_mask, node_attn=node_mask)
            loss, loss_dict = self.__loss__(self.node_mask.sigmoid(), masked_clf_logits, original_clf_logits.sigmoid(), epoch, warmup)
            loss.backward()
            optimizer.step()

        return loss, loss_dict, original_clf_logits, masked_clf_logits, self.node_mask.sigmoid().reshape(-1), None, None

    @staticmethod
    def node_attn_to_edge_attn(node_attn, edge_index):
        src_attn = node_attn[edge_index[0]]
        dst_attn = node_attn[edge_index[1]]
        edge_attn = src_attn * dst_attn
        return edge_attn
