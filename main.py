import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class DCTLayer(nn.Module):
    def __init__(self, ins, outs):
        super().__init__()
        self.W = nn.Linear(ins, outs)

    def forward(self, x):
        x = dct.dct(x)
        return self.W(x)

class DCTBlock(nn.Module):
    def __init__(self, dm):
        super().__init__()
        self.dctw = nn.Linear(dm, dm)

        self.ff = nn.Sequential(
            nn.Linear(dm, 4 * dm),
            nn.SiLU(),
            nn.Linear(4 * dm, dm)
        )
    
        self.ln1 = nn.LayerNorm(dm)
        self.ln2 = nn.LayerNorm(dm)


    def forward(self, x):
        dctx = dct.dct(x)
        x = self.ln1(x + self.dctw(dctx))
        x = self.ln2(x + self.ff(x))
        return x


def positional_encoding(max_len, dm):
    position = torch.arange(max_len).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, dm, 2) * (-math.log(100000.0) / dm))
    pe = torch.zeros(1, max_len, dm)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


class DCTNet(nn.Module):
    def __init__(self, vocab_size, dm, max_len=10000):
        super().__init__()
        self.pe = positional_encoding(10000, dm)
        self.inW = nn.Linear(vocab_size, dm)
        self.model = nn.Sequential(
            DCTBlock(dm),
            DCTBlock(dm),
            DCTBlock(dm),
            DCTBlock(dm),
            DCTBlock(dm)
        )
        self.outW = nn.Linear(dm, vocab_size)


    def forward(self, x):
        x = self.inW(x) + self.pe
        x = self.model(x)
        return F.softmax(self.outW(x), dim=-1)
        

def cross_entropy(pred, tar):
    return -torch.mean(tar * torch.log(pred))
