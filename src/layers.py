import math
import torch
from torch import nn
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=bias),
            nn.GELU(approximate="tanh"),
            *[nn.Sequential(
                nn.Linear(hidden_features, hidden_features, bias=bias),
                nn.GELU(approximate="tanh")
            ) for _ in range(num_layers - 2)],
            nn.Linear(hidden_features, out_features, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class Patchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        ph, pw = self.patch_size
        x = rearrange(x, 'b c (h ph) (w pw) -> b c (h w) ph pw', ph=ph, pw=pw)
        return x
    

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size):
        super().__init__()

        self.patch_size = patch_size
        self.patchify = Patchify(patch_size)
        self.projection = nn.Linear(in_channels * patch_size[0] * patch_size[1], emb_size)

    def forward(self, x):
        x = self.patchify(x)
        x = rearrange(x, "b c (h w) ph pw -> b (h w) (c ph pw)", ph=self.patch_size[0], pw=self.patch_size[1], h=int(x.shape[2] ** 0.5))
        x = self.projection(x)

        return x
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)

        args = t[:, None].float() * freqs[None]

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    
class ClassEmbedder(nn.Module):
    def __init__(self, num_classes: int, emb_size: int, drop_probability: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.emb_size = emb_size
        self.drop_probability = drop_probability
        self.embedding = nn.Embedding(num_classes + 1, emb_size)

    def forward(self, y):
        if self.training:
            y = torch.where(torch.rand_like(y) < self.drop_probability, y, self.num_classes)
        
        return self.embedding(y)
    
class AdaLNZeroModulation(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.activation = nn.SiLU()
        self.linear = nn.Linear(dim, dim * n * 3, bias=True)

    def init_weights(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.activation(x)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    te = TimestepEmbedder(1024)
    t = torch.arange(100).float()

    print(te(t).shape)