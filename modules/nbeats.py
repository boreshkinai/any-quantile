import torch
import torch.nn as nn
import torch.nn.functional as F


class NbeatsBlock(torch.nn.Module):
    """Fully connected residual block"""

    def __init__(self, num_layers: int, layer_width: int, size_in: int, size_out: int):
        super().__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.size_in = size_in
        self.size_out = size_out

        self.fc_layers = [torch.nn.Linear(size_in, layer_width)]
        self.fc_layers += [torch.nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)]
        self.fc_layers = torch.nn.ModuleList(self.fc_layers)
        
        self.forward_projection = torch.nn.Linear(layer_width, size_out)
        self.backward_projection = torch.nn.Linear(layer_width, size_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.fc_layers:
            h = F.relu(layer(h))
        return self.forward_projection(h), self.backward_projection(h)
    
    
class NbeatsBlockConditioned(NbeatsBlock):
    """Fully connected residual block"""

    def __init__(self, num_layers: int, layer_width: int, size_in: int, size_out: int):
        
        super().__init__(num_layers=num_layers, layer_width=layer_width, size_in=size_in, size_out=size_out)
        
        self.condition_film = torch.nn.Linear(self.layer_width, 2*self.layer_width)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.fc_layers):
            h = F.relu(layer(h))
            if i == 0:
                condition_film = self.condition_film(condition)
                condition_offset, condition_delta = condition_film[..., :self.layer_width], condition_film[..., self.layer_width:]
                h = h * (1 + condition_delta) + condition_offset
            
        return self.forward_projection(h), self.backward_projection(h)
    
    
class NBEATS(torch.nn.Module):
    def __init__(self, num_blocks: int, num_layers: int, layer_width: int, share: bool, 
                 size_in: int, size_out: int, block_class: torch.nn.Module = NbeatsBlock):
        super().__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.size_in = size_in
        self.size_out = size_out
        self.num_blocks = num_blocks
        self.share = share
        
        self.blocks = [block_class(num_layers=num_layers, 
                                   layer_width=layer_width, 
                                   size_in=size_in, size_out=size_out)]
        if self.share:
            for i in range(self.num_blocks-1):
                self.blocks.append(self.blocks[0])
        else:
            for i in range(self.num_blocks-1):
                self.blocks.append(block_class(num_layers=num_layers, 
                                               layer_width=layer_width, 
                                               size_in=size_in, size_out=size_out))
        self.blocks = torch.nn.ModuleList(self.blocks)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backcast = x
        output = 0.0
        for block in self.blocks:
            f, b = block(backcast)
            output = output + f
            backcast = backcast - b
        return output


class NBEATSAQCAT(NBEATS):
    def __init__(self, num_blocks: int, num_layers: int, layer_width: int, share: bool, size_in: int, size_out: int):
        # size_in + 1, because one position for quantile
        super().__init__(num_blocks=num_blocks, num_layers=num_layers, 
                         layer_width=layer_width, share=share, 
                         size_in=size_in+1, size_out=size_out, block_class=NbeatsBlock)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # x history time series B x T
        # q quantile specification B x Q
        # output forward prediction H horizons and Q quantiles B x H x Q
        Q = q.shape[-1]
        backcast = torch.cat([torch.repeat_interleave(x[:, None], repeats=Q, dim=1, output_size=Q), q[..., None]], dim=-1) 

        output = 0.0
        for block in self.blocks:
            f, b = block(backcast)
            output = output + f
            backcast = backcast - b

        return output.transpose(-1, -2)


class NBEATSAQOUT(NBEATS):
    def __init__(self, num_blocks: int, num_layers: int, layer_width: int, share: bool, size_in: int, size_out: int):
        # size_in + 1, because one position for quantile
        super().__init__(num_blocks=num_blocks, num_layers=num_layers, 
                         layer_width=layer_width, share=share, 
                         size_in=size_in, size_out=size_out, block_class=NbeatsBlock)
        
        self.q_block = NbeatsBlockConditioned(layer_width=layer_width, size_in=size_in, 
                                              size_out=size_out, num_layers=num_layers)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # x history time series B x T
        # q quantile specification B x Q
        # output forward prediction H horizons and Q quantiles B x H x Q

        backcast = x
        output = 0.0
        for block in self.blocks:
            f, b = block(backcast)
            output = output + f
            backcast = backcast - b

        Q = q.shape[-1]
        backcast = torch.repeat_interleave(backcast[:, None], repeats=Q, dim=1, output_size=Q)
        q = torch.repeat_interleave(q[..., None], repeats=self.layer_width, dim=-1, output_size=self.layer_width)
        
        f, b = self.q_block(backcast, condition=q)
        output = output[:, None] + f
        
        return output.transpose(-1, -2)


class NBEATSAQFILM(NBEATS):
    def __init__(self, num_blocks: int, num_layers: int, layer_width: int, share: bool, size_in: int, size_out: int):
        # size_in + 1, because one position for quantile
        super().__init__(num_blocks=num_blocks, num_layers=num_layers, 
                         layer_width=layer_width, share=share, 
                         size_in=size_in, size_out=size_out, block_class=NbeatsBlockConditioned)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # x history time series B x T
        # q quantile specification B x Q
        # output forward prediction H horizons and Q quantiles B x H x Q

        Q = q.shape[-1]
        backcast = torch.repeat_interleave(x[:, None], repeats=Q, dim=1, output_size=Q)
        q = torch.repeat_interleave(q[..., None], repeats=self.layer_width, dim=-1, output_size=self.layer_width)

        output = 0.0
        for i, block in enumerate(self.blocks):
            f, b = block(x=backcast, condition=q)
            output = output + f
            backcast = backcast - b

        return output.transpose(-1, -2)
