from torchmetrics import Metric
from losses import MQLoss
import torch


def _divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary funtion to handle divide by 0
    """
    div = torch.nan_to_num(a / b, posinf=0.0, nan=0.0)
    return div


class Coverage(Metric):
    def __init__(self, dist_sync_on_step=False, level=0.95):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.level_low = (1.0-level)/2
        self.level_high = 1.0 - self.level_low
        self.level = level

    def add_evaluation_quantiles(self, quantiles: torch.Tensor):
        quantiles_metric = torch.Tensor([(1 - (1-self.level)/2), (1-self.level)/2])
        quantiles_metric = torch.repeat_interleave(quantiles_metric[None], repeats=quantiles.shape[0], dim=0)
        quantiles_metric = quantiles_metric.to(quantiles)
        return torch.cat([quantiles, quantiles_metric], dim=-1)

    def update(self, preds: torch.Tensor, target: torch.Tensor, q: torch.Tensor) -> None:
        """ Compute mutli-quantile loss function
        
        :param input: Bx..xQ tensor of predicted values, Q is the number of quantiles
        :param target: Bx..x1, or Bx.. tensor of target values
        :param q: Bx..xQ tensor of quantiles telling which quantiles input predictions correspond to
        :return: value if mutli-quantile loss function
        """

        if target.dim() != preds.dim():
            target = target[..., None]

        num_high = (q==self.level_high).sum(dim=-1, keepdims=True)
        num_low = (q==self.level_low).sum(dim=-1, keepdims=True)
        preds_high = (preds * (q==self.level_high)).sum(dim=-1, keepdims=True) / num_high
        preds_low = (preds * (q==self.level_low)).sum(dim=-1, keepdims=True) / num_low
                
        self.numerator += ((target < preds_high) * (target >= preds_low)).sum()
        self.denominator += torch.numel(target)

    def compute(self):
        return self.numerator / (self.denominator)
        

class CRPS(Metric):
    def __init__(self, dist_sync_on_step=False, horizon=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.horizon = horizon
        self.mqloss = MQLoss()

    def update(self, preds: torch.Tensor, target: torch.Tensor, q: torch.Tensor) -> None:
        """ Compute mutli-quantile loss function
        
        :param input: BxHxQ tensor of predicted values, Q is the number of quantiles
        :param target: BxHx1, or BxH tensor of target values
        :param q: BxHxQ or Bx1xQ tensor of quantiles telling which quantiles input predictions correspond to
        :return: value if mutli-quantile loss function
        """
        
        if self.horizon is None:
            self.numerator += self.mqloss(input=preds, target=target, q=q) * torch.numel(preds)
            self.denominator += torch.numel(preds)
        else:
            self.numerator += self.mqloss(input=preds[:, self.horizon], target=target[:, self.horizon], 
                                          q=q[:, self.horizon]) * torch.numel(preds[:, self.horizon])
            self.denominator += torch.numel(preds[:, self.horizon])

    def compute(self):
        return 2*(self.numerator / self.denominator)

    
class MAPE(Metric):
    def __init__(self, dist_sync_on_step=False, horizon=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("smape", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("nsamples", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.horizon = horizon

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        if self.horizon is None:
            smape = _divide_no_nan(torch.abs(target - preds), torch.abs(target))
            self.smape += smape.sum()
            self.nsamples += torch.numel(smape)
        else:
            smape = _divide_no_nan(torch.abs(target[:, self.horizon] - preds[:, self.horizon]), 
                                   torch.abs(target[:, self.horizon]))
            self.smape += smape.sum()
            self.nsamples += torch.numel(target[:, self.horizon])

    def compute(self):
        return 100*(self.smape / self.nsamples)
    

class SMAPE(Metric):
    def __init__(self, dist_sync_on_step=False, horizon=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("smape", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("nsamples", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.horizon = horizon

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        if self.horizon is None:
            smape = 2 * _divide_no_nan(torch.abs(target - preds), torch.abs(target) + torch.abs(preds))
            self.smape += smape.sum()
            self.nsamples += torch.numel(smape)
        else:
            smape = 2 * _divide_no_nan(torch.abs(target[:, self.horizon] - preds[:, self.horizon]), 
                                       torch.abs(target[:, self.horizon]) + torch.abs(preds[:, self.horizon]))
            self.smape += smape.sum()
            self.nsamples += torch.numel(target[:, self.horizon])

    def compute(self):
        return 100*(self.smape / self.nsamples)[0]

    
class WAPE(Metric):
    def __init__(self, dist_sync_on_step=False, horizon=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.horizon = horizon

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        if self.horizon is None:
            self.numerator += torch.abs(target - preds).sum()
            self.denominator += target.sum()
        else:
            self.numerator += torch.abs(target[:, self.horizon] - preds[:, self.horizon]).sum()
            self.denominator += target[:, self.horizon].sum()

    def compute(self):
        return 100*(self.numerator / self.denominator)[0]
