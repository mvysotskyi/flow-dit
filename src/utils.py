import torch
from safetensors.torch import load_file as load_sft

from src.dit import Dit
from src.config import DitConfig
from src.autoencoder import AutoEncoder, AutoEncoderParams


def load_dit(dit_config: DitConfig, checkpoint_path: str, device) -> Dit:
    with torch.device("meta" if torch.cuda.is_available() and checkpoint_path else device):
        dit = Dit(dit_config)

    if checkpoint_path is not None:
        dit.load_state_dict(
            load_sft(checkpoint_path, device=device),
            strict=False, assign=True
        )

    return dit

def load_ae(ae_config: AutoEncoderParams, checkpoint_path: str, device) -> AutoEncoder:
    with torch.device("meta" if torch.cuda.is_available() and checkpoint_path else device):
        ae = AutoEncoder(ae_config)

    if checkpoint_path is not None:
        ae.load_state_dict(
            load_sft(checkpoint_path, device=device),
            strict=False, assign=True
        )

    return ae