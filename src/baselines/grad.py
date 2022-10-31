# https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/base_cam.py

import numpy as np
import torch
import torch.nn as nn


class Grad(nn.Module):
    def __init__(self, clf, signal_class, criterion, config) -> None:
        super().__init__()
        self.clf = clf
        self.target_layers = [clf.model.convs[-1]]
        self.criterion = criterion
        self.gradgeo = config.get('gradgeo', False)
        self.device = next(self.parameters()).device

        assert signal_class is not None
        self.signal_class = signal_class

    def start_tracking(self):
        self.activations_and_grads = ActivationsAndGradients(self.clf, self.target_layers)

    def get_cam(self, activations: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
        assert activations.min() >= 0.0  # rectified
        cam = (grads * activations).sum(1)
        return cam

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.stack(cam_per_target_layer, axis=1)
        result = np.mean(cam_per_target_layer, axis=1)
        return result

    def __loss__(self, attn, clf_logits, clf_labels, epoch, warmup):
        assert warmup
        pred_loss = self.criterion(clf_logits, clf_labels.float())
        if warmup:
            return pred_loss, {'loss': pred_loss.item(), 'pred': pred_loss.item()}

    def forward_pass(self, data, epoch, warmup, **kwargs):
        if warmup:
            clf_logits = self.clf(data)
            loss, loss_dict = self.__loss__(None, clf_logits, data.y, epoch, warmup)
            return loss, loss_dict, clf_logits, None, None, None, None

        self.clf.eval()
        if self.gradgeo:
            data.pos.requires_grad = True

        original_clf_logits = self.activations_and_grads(data)
        masked_clf_logits = original_clf_logits
        targets = [BinaryClassifierOutputTarget(self.signal_class)] * original_clf_logits.shape[0]

        self.clf.zero_grad()
        loss = sum([target(output) for target, output in zip(targets, original_clf_logits)])
        loss.backward(retain_graph=True)
        loss_dict = {'loss': loss.item(), 'pred': loss.item()}

        pred_dir = None
        if self.gradgeo:
            node_weights = data.pos.grad.norm(dim=1, p=2)
            pred_dir = data.pos.grad[:, :2]
        else:
            cam_per_layer = self.compute_cam_per_layer()
            node_weights = torch.tensor(self.aggregate_multi_layers(cam_per_layer))
        return loss, loss_dict, original_clf_logits, masked_clf_logits, node_weights.reshape(-1), pred_dir, None

    def compute_cam_per_layer(self) -> np.ndarray:
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam(layer_activations, layer_grads)
            cam_per_target_layer.append(cam)
        return cam_per_target_layer

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


class BinaryClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if self.category == 1:
            sign = 1
        else:
            sign = -1
        return model_output * sign


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform=None):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        if isinstance(output, tuple):  # egnn outputs a tuple
            output = output[0]

        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if isinstance(output, tuple):  # egnn outputs a tuple
            output = output[0]

        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()
