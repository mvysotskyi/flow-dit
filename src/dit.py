import torch
from torch import nn
from einops import rearrange

from diffusers.models.embeddings import get_2d_sincos_pos_embed

from src.config import DitConfig, dit_configs
from src.layers import PatchEmbedding, TimestepEmbedder, ClassEmbedder, MLP, AdaLNZeroModulation


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DitBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = MLP(hidden_size, hidden_size * 4, hidden_size, num_layers=2)
        self.modulation = AdaLNZeroModulation(hidden_size, 2)

    def forward(self, x, c):
        alpha1, gamma1, beta1, alpha2, gamma2, beta2 = self.modulation(c).chunk(6, dim=1)
        
        x_norm1_modulated = modulate(self.norm1(x), beta1, gamma1)
        x = x + alpha1.unsqueeze(1) * self.attn(x_norm1_modulated, x_norm1_modulated, x_norm1_modulated)[0]

        x_norm2_modulated = modulate(self.norm2(x), beta2, gamma2)
        x = x + alpha2.unsqueeze(1) * self.mlp(x_norm2_modulated)

        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, patch_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.modulation = AdaLNZeroModulation(hidden_size, 1)
        self.linear = nn.Linear(hidden_size, out_channels * patch_size * patch_size)

    def init_weights(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        _, gamma, beta = self.modulation(c).chunk(3, dim=1)
        
        x_norm_modulated = modulate(self.norm(x), beta, gamma)
        x = self.linear(x_norm_modulated)

        return x


class Dit(nn.Module):
    def __init__(self, config: DitConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbedding(config.in_channels, (config.patch_size, config.patch_size), config.hidden_size)
        self.timestep_embedder = TimestepEmbedder(config.hidden_size, config.frequency_embedding_size)
        self.label_embedder = ClassEmbedder(config.num_classes, config.hidden_size, drop_probability=config.condition_dropout_prob)

        self.final_modulation = AdaLNZeroModulation(config.hidden_size, 1)
        self.final_layer = FinalLayer(config.hidden_size, config.out_channels, config.patch_size)

        self.dit_blocks = nn.ModuleList([DitBlock(config.hidden_size, config.num_attention_heads) for _ in range(config.depth)])

        self.pos_embeddings = get_2d_sincos_pos_embed(
            embed_dim=config.hidden_size, 
            grid_size=self.config.input_size // self.config.patch_size, 
            output_type="pt"
        ).to(torch.float32)
        self.pos_embeddings = nn.Parameter(self.pos_embeddings, requires_grad=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor):
        t_emb = self.timestep_embedder(t)
        c = self.label_embedder(labels)
        c = c + t_emb

        x = self.patch_embed(x)
        x = x + self.pos_embeddings

        for block in self.dit_blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = rearrange(
            x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", 
            h=self.config.input_size // self.config.patch_size, 
            ph=self.config.patch_size, 
            pw=self.config.patch_size
        )

        return x
    

if __name__ == "__main__":
    dit = Dit(dit_configs["dit_base_256_2_birds"])
    
    print(sum(p.numel() for p in dit.parameters()))