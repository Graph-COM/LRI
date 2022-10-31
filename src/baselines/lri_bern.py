import torch
import torch.nn as nn
from torch_scatter import scatter
import numpy as np


class LRIBern(nn.Module):

    def __init__(self, clf, extractor, criterion, config):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device

        self.pred_loss_coef = config['pred_loss_coef']
        self.info_loss_coef = config['info_loss_coef']
        self.temperature = config['temperature']

        self.final_r = config['final_r']
        self.decay_interval = config['decay_interval']
        self.decay_r = config['decay_r']
        self.init_r = config['init_r']

        self.attn_constraint = config['attn_constraint']

    def __loss__(self, attn, clf_logits, clf_labels, epoch, warmup):
        pred_loss = self.criterion(clf_logits, clf_labels.float())
        if warmup:
            return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}

        r = self.get_r(epoch)
        info_loss = (attn * torch.log(attn/r + 1e-6) + (1 - attn) * torch.log((1 - attn)/(1 - r + 1e-6) + 1e-6)).mean()

        pred_loss = self.pred_loss_coef * pred_loss
        info_loss = self.info_loss_coef * info_loss

        loss = pred_loss + info_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item(), 'r': r}
        return loss, loss_dict

    def forward_pass(self, data, epoch, warmup, do_sampling):
        if warmup:
            clf_logits = self.clf(data)
            loss, loss_dict = self.__loss__(None, clf_logits, data.y, epoch, warmup)
            return loss, loss_dict, clf_logits, None, None, None, None

        (emb, pool_out_lig), edge_index = self.clf.get_emb(data)
        node_attn_log_logits = self.extractor(emb, batch=data.batch, pool_out_lig=pool_out_lig)

        if self.attn_constraint == 'smooth_min':
            node_attn_log_logits = scatter(node_attn_log_logits[edge_index[1]].reshape(-1), edge_index[0], reduce='min').reshape(-1, 1)
            node_attn_log_logits = scatter(node_attn_log_logits[edge_index[1]].reshape(-1), edge_index[0], reduce='min').reshape(-1, 1)
        else:
            assert self.attn_constraint == 'none'

        node_attn = self.sampling(node_attn_log_logits, do_sampling)
        edge_attn = self.node_attn_to_edge_attn(node_attn, edge_index)
        masked_clf_logits = self.clf(data, edge_attn=edge_attn, node_attn=node_attn)
        original_clf_logits = self.clf(data)

        loss, loss_dict = self.__loss__(node_attn_log_logits.sigmoid(), masked_clf_logits, data.y, epoch, warmup)
        return loss, loss_dict, original_clf_logits, masked_clf_logits, node_attn.reshape(-1), None, None

    def get_r(self, current_epoch):
        r = self.init_r - current_epoch // self.decay_interval * self.decay_r
        if r < self.final_r:
            r = self.final_r
        return r

    def sampling(self, attn_log_logits, do_sampling):
        if do_sampling:
            random_noise = torch.empty_like(attn_log_logits).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            attn_bern = ((attn_log_logits + random_noise) / self.temperature).sigmoid()
        else:
            attn_bern = (attn_log_logits).sigmoid()
        return attn_bern

    @staticmethod
    def node_attn_to_edge_attn(node_attn, edge_index):
        src_attn = node_attn[edge_index[0]]
        dst_attn = node_attn[edge_index[1]]
        edge_attn = src_attn * dst_attn
        return edge_attn
