# https://github.com/asgsaeid/PointMask/blob/main/model_cls.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class PointMask(nn.Module):

    def __init__(self, clf, extractor, criterion, config):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device

        self.t = config['t']
        self.pred_loss_coef = config['pred_loss_coef']
        self.kl_loss_coef = config['kl_loss_coef']

    def __loss__(self, mu, log_var, clf_logits, clf_labels, epoch, warmup, batch):
        pred_loss = self.criterion(clf_logits, clf_labels.float())
        if warmup:
            return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}

        kl_loss = (- 0.5 * scatter_add(1.0 + log_var - mu.pow(2) - log_var.exp(), batch.reshape(-1, 1), dim=0)).mean()
        # kl_batch = - 0.5 * K.sum(1. + log_var - K.square(mu) - K.exp(log_var), axis=-1)

        pred_loss = self.pred_loss_coef * pred_loss
        kl_loss = self.kl_loss_coef * kl_loss

        loss = pred_loss + kl_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'kl': kl_loss.item()}
        return loss, loss_dict

    def maskrelu(self, x):
        x = torch.sigmoid(x)
        inv_msk = F.relu(x - self.t).clamp(max=1.0)
        return inv_msk

    def forward_pass(self, data, epoch, warmup, do_sampling):
        if warmup:
            clf_logits = self.clf(data)
            loss, loss_dict = self.__loss__(None, None, clf_logits, data.y, epoch, warmup, None)
            return loss, loss_dict, clf_logits, None, None, None, None

        (emb, pool_out_lig), _ = self.clf.get_emb(data)
        U = self.extractor(emb, batch=data.batch, pool_out_lig=pool_out_lig)

        z_mu = U[:, [0]]
        z_log_var = U[:, [1]]
        z_sigma = torch.exp(z_log_var / 2)

        if do_sampling:
            eps = torch.randn_like(z_sigma)
            z = z_mu + z_sigma * eps
        else:
            z = z_mu

        node_attn = self.maskrelu(z)
        data.pos = data.pos * node_attn
        masked_clf_logits = self.clf(data)
        original_clf_logits = masked_clf_logits

        loss, loss_dict = self.__loss__(z_mu, z_log_var, masked_clf_logits, data.y, epoch, warmup, data.batch)
        return loss, loss_dict, original_clf_logits, masked_clf_logits, node_attn.reshape(-1), None, None
