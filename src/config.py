from dataclasses import dataclass
from src.autoencoder import AutoEncoderParams

@dataclass
class DitConfig:
    input_size: int
    patch_size: int
    hidden_size: int
    depth: int
    num_attention_heads: int
    patch_size: int
    in_channels: int
    out_channels: int
    num_classes: int
    condition_dropout_prob: float
    frequency_embedding_size: int


dit_configs = {
    "dit_base_256_2_birds": DitConfig(
        input_size=32,
        patch_size=2,
        hidden_size=768,
        depth=12,
        num_attention_heads=12,
        in_channels=4,
        out_channels=4,
        num_classes=200,
        condition_dropout_prob=0.15,
        frequency_embedding_size=256
    ),
}

ae_configs = {
    "ae_256_ch16": AutoEncoderParams(
        resolution=256,
        in_channels=3,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )
}