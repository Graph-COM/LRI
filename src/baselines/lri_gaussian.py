import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max


class LRIGaussian(nn.Module):

    def __init__(self, clf, extractor, criterion, config):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.criterion = criterion
        self.device = next(self.parameters()).device

        self.pred_loss_coef = config['pred_loss_coef']
        self.info_loss_coef = config['info_loss_coef']
        self.dim = config['covar_dim']
        self.attn_constraint = config['attn_constraint']

    @staticmethod
    def kl(pred_sigma, reg_sigma):
        first_term = torch.log(reg_sigma.det() / (abs(pred_sigma.det()) + 1e-6))
        second_term = -reg_sigma.shape[0]
        third_term = batch_trace(torch.inverse(reg_sigma) @ pred_sigma)
        return (first_term + second_term + third_term) / 2

    def __loss__(self, pred_sigma, clf_logits, clf_labels, epoch, warmup):
        pred_loss = self.criterion(clf_logits, clf_labels.float())
        if warmup:
            return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}

        reg_sigma = self.get_reg_sigma(epoch)
        info_loss = self.kl(pred_sigma, reg_sigma).mean()

        pred_loss = self.pred_loss_coef * pred_loss
        info_loss = self.info_loss_coef * info_loss

        loss = pred_loss + info_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item(), 'sig': reg_sigma.det().item()}
        return loss, loss_dict

    def forward_pass(self, data, epoch, warmup, do_sampling):
        if warmup:
            clf_logits = self.clf(data)
            loss, loss_dict = self.__loss__(None, clf_logits, data.y, epoch, warmup)
            return loss, loss_dict, clf_logits, None, None, None, None

        (emb, pool_out_lig), edge_index = self.clf.get_emb(data)
        U = self.extractor(emb, batch=data.batch, pool_out_lig=pool_out_lig)

        sig1 = F.softplus(U[:, [0]]).clamp(1e-6, 1e6)
        sig2 = F.softplus(U[:, [1]]).clamp(1e-6, 1e6)

        U = U[:, 2:].reshape(emb.shape[0], self.dim, self.dim)
        pred_sigma = sig1.reshape(-1, 1, 1) * U @ U.transpose(1, 2) + sig2.reshape(-1, 1, 1) * torch.eye(self.dim, device=self.device).reshape(-1, self.dim, self.dim)
        pred_sigma = self.smooth_min(edge_index, pred_sigma, U, sig1, sig2) if self.attn_constraint == 'smooth_min' else pred_sigma

        node_noise = self.sampling(U, do_sampling, sig1, sig2)
        masked_clf_logits = self.clf(data, node_noise=node_noise)
        original_clf_logits = self.clf(data)

        loss, loss_dict = self.__loss__(pred_sigma, masked_clf_logits, data.y, epoch, warmup)

        return loss, loss_dict, original_clf_logits, masked_clf_logits, -(pred_sigma.det()).reshape(-1), pred_sigma, node_noise

    def sampling(self, U, do_sampling, sig1, sig2):
        if do_sampling:
            epsilon_1 = torch.randn((U.shape[0], U.shape[2], 1), device=self.device)
            epsilon_2 = torch.randn((U.shape[0], U.shape[2], 1), device=self.device)
            z = sig1.sqrt() * (U @ epsilon_1).squeeze(-1) + sig2.sqrt() * epsilon_2.squeeze(-1)
        else:
            z = None
        return z

    def get_reg_sigma(self, epoch):
        reg_var = torch.tensor([1.0]*self.dim, device=self.device) * 1.0
        reg_sigma = torch.diag_embed(reg_var)
        return reg_sigma

    @staticmethod
    def smooth_min(edge_index, pred_sigma, U, sig1, sig2):
        det = torch.det(pred_sigma)
        _, min_neighbour_idx = scatter_max(det[edge_index[1]], edge_index[0])
        U = U[edge_index[1]][min_neighbour_idx]
        sig1 = sig1[edge_index[1]][min_neighbour_idx]
        sig2 = sig2[edge_index[1]][min_neighbour_idx]
        pred_sigma = pred_sigma[edge_index[1]][min_neighbour_idx]

        det = torch.det(pred_sigma)
        _, min_neighbour_idx = scatter_max(det[edge_index[1]], edge_index[0])
        U = U[edge_index[1]][min_neighbour_idx]
        sig1 = sig1[edge_index[1]][min_neighbour_idx]
        sig2 = sig2[edge_index[1]][min_neighbour_idx]
        pred_sigma = pred_sigma[edge_index[1]][min_neighbour_idx]
        return pred_sigma

def batch_trace(batch_sig):
    return batch_sig.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
