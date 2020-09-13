import torch
from torch import nn

class CELDice:
    def __init__(self, dice_weight=0,num_classes=1):
        self.nll_loss = nn.NLLLoss()
        self.jaccard_weight = dice_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
       loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
       if self.jaccard_weight:
           eps = 1e-15
           for cls in range(self.num_classes):
               jaccard_target = (targets == cls).float()
               jaccard_output = outputs[:, cls].exp()
               intersection = (jaccard_output * jaccard_target).sum()
               union = jaccard_output.sum() + jaccard_target.sum()
               loss -= torch.log((2*intersection + eps) / (union + eps)) * self.jaccard_weight
       return loss

