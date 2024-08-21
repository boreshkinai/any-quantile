from torchmetrics import Metric
import torch


class WBIAS(Metric):
    def __init__(self, compute_on_step=True, dist_sync_on_step=False, horizon=None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.horizon = horizon

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        if self.horizon is None:
            self.numerator += (preds - target).sum()
            self.denominator += target.sum()
        else:
            self.numerator += (preds[:, self.horizon] - target[:, self.horizon]).sum()
            self.denominator += target[:, self.horizon].sum()

    def compute(self):
        return 100*(self.numerator / self.denominator)[0]