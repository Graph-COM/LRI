# https://github.com/pyg-team/pytorch_geometric/blob/72eb1b38f60124d4d700060a56f7aa9a4adb7bb0/torch_geometric/nn/models/pg_explainer.py

import torch
import torch.nn as nn
from torch_scatter import scatter_mean


class BernMaskP(nn.Module):

    def __init__(self, clf, extractor, criterion, config):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device

        self.pred_loss_coef = config['pred_loss_coef']
        self.size_loss_coef = config['size_loss_coef']
        self.mask_ent_loss_coef = config['mask_ent_loss_coef']
        self.temp_tuple = config['temp']
        self.epochs = config['epochs']

    def __loss__(self, mask, clf_logits, clf_labels, epoch, warmup, batch):
        pred_loss = self.criterion(clf_logits, clf_labels.float())
        if warmup:
            return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}

        # size_loss = attn.sum()
        size_loss = mask.mean()  # take mean makes it easier to tune coef for different datasets
        mask_ent_reg = -mask * (mask + 1e-10).log() - (1 - mask) * (1 - mask + 1e-10).log()
        # mask_ent_loss = mask_ent_reg.mean() if batch is None else scatter_mean(mask_ent_reg, batch.reshape(-1, 1)).sum()
        mask_ent_loss = mask_ent_reg.mean() if batch is None else scatter_mean(mask_ent_reg, batch.reshape(-1, 1)).mean()  # take mean makes it easier to tune coef for different datasets

        pred_loss = self.pred_loss_coef * pred_loss
        size_loss = self.size_loss_coef * size_loss
        mask_ent_loss = self.mask_ent_loss_coef * mask_ent_loss

        loss = pred_loss + size_loss + mask_ent_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'size': size_loss.item(), 'ent': mask_ent_loss.item()}
        return loss, loss_dict

    def forward_pass(self, data, epoch, warmup, do_sampling):
        if warmup:
            clf_logits = self.clf(data)
            loss, loss_dict = self.__loss__(None, clf_logits, data.y, epoch, warmup, None)
            return loss, loss_dict, clf_logits, None, None, None, None

        (emb, pool_out_lig), edge_index = self.clf.get_emb(data)
        node_mask_log_logits = self.extractor(emb, batch=data.batch, pool_out_lig=pool_out_lig)

        node_mask = self.sampling(node_mask_log_logits, do_sampling, epoch)
        edge_mask = self.node_attn_to_edge_attn(node_mask, edge_index)

        original_clf_logits = self.clf(data)
        masked_clf_logits = self.clf(data, edge_attn=edge_mask, node_attn=node_mask)

        loss, loss_dict = self.__loss__(node_mask_log_logits.sigmoid(), masked_clf_logits, original_clf_logits.sigmoid(), epoch, warmup, data.batch)
        return loss, loss_dict, original_clf_logits, masked_clf_logits, node_mask.reshape(-1), None, None

    def get_temp(self, e: int) -> float:
        temp = self.temp_tuple
        return temp[0] * ((temp[1] / temp[0])**(e / self.epochs))

    def sampling(self, attn_log_logits, do_sampling, epoch):
        if do_sampling:
            random_noise = torch.empty_like(attn_log_logits).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)

            temp = self.get_temp(epoch)
            attn_bern = ((attn_log_logits + random_noise) / temp).sigmoid()
        else:
            attn_bern = (attn_log_logits).sigmoid()
        return attn_bern

    @staticmethod
    def node_attn_to_edge_attn(node_attn, edge_index):
        src_attn = node_attn[edge_index[0]]
        dst_attn = node_attn[edge_index[1]]
        edge_attn = src_attn * dst_attn
        return edge_attn
