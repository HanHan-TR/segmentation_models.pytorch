from segmentation_models_pytorch.losses import (DiceLoss,
                                                SoftCrossEntropyLoss,
                                                FocalLoss,
                                                TverskyLoss,
                                                LovaszLoss)
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self,
                 losses=['dice', 'focal'],
                 losses_weights=None,
                 alpha=0.5,
                 mode='multiclass'):
        super().__init__()
        # self.tversky_loss = TverskyLoss(mode=mode, reduction=reduction)
        # self.lovasz_loss = LovaszLoss(mode=mode, reduction=reduction)
        self.loss_dict = {'dice': DiceLoss(mode=mode),
                          'focal': FocalLoss(mode=mode, alpha=alpha),
                          'bce': SoftCrossEntropyLoss()}

        if losses_weights is None:
            losses_weights = [1.0] * len(losses)

        assert len(losses) == len(losses_weights)

        self.losses = losses
        self.losses_weights = losses_weights

    def forward(self, logits, masks):
        loss = 0
        for loss_name, weight in zip(self.losses, self.losses_weights):
            loss += weight * self.loss_dict[loss_name](logits, masks)

        return loss
