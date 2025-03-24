import torch
from safetensors.torch import load_file as load_sft

from src.autoencoder import AutoEncoder, AutoEncoderParams


def load_ae(ae_config: AutoEncoderParams, checkpoint_path: str) -> AutoEncoder:
    with torch.device("meta" if torch.cuda.is_available() else "cpu"):
        ae = AutoEncoder(ae_config)

    ae.load_state_dict(
        load_sft(checkpoint_path, device="cuda"),
        strict=False, assign=True
    )
    return ae