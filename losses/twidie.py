from torch.nn.modules.loss import _Loss
import torch


class TwidieLoss(_Loss):
    
#     https://towardsdatascience.com/tweedie-loss-function-for-right-skewed-data-2c5ca470678f
    
    __constants__ = ['reduction']

    def __init__(self, p: float = 1.5, reduction: str = 'mean', eps: float = 1e-3) -> None:
        super().__init__(reduction=reduction)
        self.p = p
        self.eps = eps
        assert p > 1, "p must be greater than 1"
        assert p < 2, "p must be smaller than 2"

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        
        input = torch.where(input > self.eps, input, self.eps*torch.ones_like(input))
        loss = - target * torch.pow(input, 1 - self.p) / (1 - self.p) + torch.pow(input, 2 - self.p) / (2 - self.p)
        
        if self.reduction == 'none':
            return loss
        
        if weights is None:
            return torch.mean(loss)
        else:
            return torch.sum(loss * weights) / max(torch.sum(weights), self.eps)
        