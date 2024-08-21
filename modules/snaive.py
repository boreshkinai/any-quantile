import torch
    
    
class SNAIVE(torch.nn.Module):
    def __init__(self, size_in: int, size_out: int, lag: int):
        super(SNAIVE, self).__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.lag = lag
        self.dummy_layer = torch.nn.Linear(size_in, 1)
        
        assert self.lag >= size_out
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dummy = self.dummy_layer(x)
        naive = x[:, -self.lag:] if (self.lag-self.size_out) == 0 else x[:, -self.lag:-self.lag+self.size_out]
        return naive + 0.0*dummy
    