import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torch_dct as dct


from transformers import BertTokenizer


DATA_PATH = "PATH_TO_TEXT"


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

        self.ff_dct = nn.Sequential(
            nn.Linear(dm, 4 * dm),
            nn.SiLU(),
            nn.Linear(4 * dm, dm)
        )
        self.ff_idct = nn.Sequential(
            nn.Linear(dm, 4 * dm),
            nn.SiLU(),
            nn.Linear(4 * dm, dm)
        )
        self.ff_linear = nn.Sequential(
            nn.Linear(dm, 4 * dm),
            nn.SiLU(),
            nn.Linear(4 * dm, dm)
        )
    
        self.ln1 = nn.LayerNorm(dm)
        self.ln2 = nn.LayerNorm(dm)


    def forward(self, x):
        # DCT along timestep
        dctx = torch.transpose(dct.dct(torch.transpose(x, 1, 2)), 1, 2)
        dctx = dctx + self.ff_dct(dctx)
        inv_dctx = torch.transpose(dct.idct(torch.transpose(dctx, 1, 2)), 1, 2)
        x = self.ln1(x + self.ff_idct(inv_dctx))
        x = self.ln2(x + self.ff_linear(x))
        return x


def positional_encoding(max_len, dm):
    position = torch.arange(max_len).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, dm, 2) * (-math.log(100000.0) / dm))
    pe = torch.zeros(1, max_len, dm)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


class DCTNet(nn.Module):
    def __init__(self, vocab_size, dm, num_blocks=5, max_len=10000):
        super().__init__()
        self.pe = positional_encoding(10000, dm)
        self.embed = nn.Embedding(vocab_size, dm)
        layers = [DCTBlock(dm) for _ in range(num_blocks)]
        self.model = nn.Sequential(*layers)
        self.outW = nn.Linear(dm, vocab_size)

    def forward(self, x):
        x = self.embed(x) + self.pe[:, :x.shape[-1]]
        x = self.model(x)
        return F.softmax(self.outW(x), dim=-1)
        

def cross_entropy(pred, tar):
    return -torch.mean(tar * torch.log(pred))


def text_to_tokens(text):
    tokens = np.frombuffer(bytes(text, "utf-8"), np.uint8)
    return tokens.astype(np.int32)


if __name__=="__main__":
    with open(DATA_PATH, "r") as f:
        data = f.read()
    data = text_to_tokens(data)

    batch_size = 10
    vocab_size = 256

    model = DCTNet(vocab_size, 512, num_blocks=5)
    opt = optim.Adam(model.parameters())

    for i in range(1):
        opt.zero_grad()
        batch_len = np.random.randint(20, 5000)
        batch_starts = np.random.randint(0, data.shape[0] - batch_len - 1, size=batch_size)

        batch_x = np.array([data[start:start+batch_len] for start in batch_starts])
        y_indices = data[batch_starts+batch_len]
        batch_y = np.zeros((batch_size, vocab_size))
        batch_y[np.arange(batch_size), y_indices] = 1

        batch_x = torch.from_numpy(batch_x)
        batch_y = torch.from_numpy(batch_y)

        y = model(batch_x)[:, 0, :]

        loss = cross_entropy(y, batch_y)

        loss.backward()

        opt.step()

        if i % 100 == 0:
            print("Loss: %.4f"%loss)



        




