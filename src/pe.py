import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len = 512):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype = torch.float32).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (- torch.log(torch.tensor(10000.0)) / embed_size))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, x.size(1), :].to(x.device)
    

pe = PositionalEncoding(embed_size = 512, max_len = 512)
pe.forward(x = torch.randn(1, 128, 512))

