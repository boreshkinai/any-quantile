from torch.nn.modules.loss import _Loss
import torch


class MQLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(MQLoss, self).__init__(reduction=reduction)
        assert reduction == 'mean', f"Reduction method {reduction} is not implemented"

    def forward(self, input: torch.Tensor, target: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """ Compute mutli-quantile loss function
        
        :param input: Bx...xQ tensor of predicted values, Q is the number of quantiles
        :param target: Bx...x1, B or Bx...x1 tensor of target values
        :param q: Bx1x...xQ tensor of quantiles telling which quantiles input predictions correspond to
        :return: value if mutli-quantile loss function
        """
        if target.dim() != input.dim():
            target = target[..., None]
            
        pinball = torch.where(target >= input, q*(target - input), (1-q)*(input-target))
        
        return torch.mean(pinball)


class MQNLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(MQNLoss, self).__init__(reduction=reduction)
        assert reduction == 'mean', f"Reduction method {reduction} is not implemented"

    def forward(self, input: torch.Tensor, target: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """ Compute mutli-quantile loss function
        
        :param input: Bx...xQ tensor of predicted values, Q is the number of quantiles
        :param target: Bx...x1, B or Bx...x1 tensor of target values
        :param q: Bx1x...xQ tensor of quantiles telling which quantiles input predictions correspond to
        :return: value if mutli-quantile loss function
        """
        if target.dim() != input.dim():
            target = target[..., None]

        denominator = target.clone()
        denominator[denominator < 1] = 1
            
        pinball = torch.where(target >= input, 
                              q*(target - input) / denominator, 
                              (1-q)*(input-target) / denominator)
        
        return torch.mean(pinball)


class PinballLoss(_Loss):
    
    __constants__ = ['reduction']

    def __init__(self, tau: float = 0.5, reduction: str = 'mean', eps: float = 1e-6) -> None:
        super(PinballLoss, self).__init__(reduction=reduction)
        self.tau = tau
        self.eps = eps
        assert reduction == 'mean', f"Reduction method {reduction} is not implemented"

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        
        pinball = torch.where(target >= input,
                              self.tau*(target - input), 
                              (1-self.tau)*(input-target))
        
        if weights is None:
            return torch.mean(pinball)
        else:
            return torch.sum(pinball * weights) / max(torch.sum(weights), self.eps)
        
        
class PinballMape(PinballLoss):
    
    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        denominator = target.clone()
        denominator[denominator < 1] = 1
        pinball = 200*torch.where(target >= input, 
                                  self.tau*(target - input) / denominator, 
                                  (1-self.tau)*(input-target) / denominator)
        
        if weights is None:
            return torch.mean(pinball)
        else:
            return torch.sum(pinball * weights) / max(torch.sum(weights), self.eps)
        
        
