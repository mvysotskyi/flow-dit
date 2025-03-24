import os
import argparse

import torch
from PIL import Image

from src.dit import Dit
from src.autoencoder import AutoEncoder, AutoEncoderParams
from src.config import DitConfig, dit_configs, ae_configs
from src.utils import load_ae


def generate(
    dit: Dit,
    ae: AutoEncoder, 
    labels: list[int], 
    guidance_scale: float = 1.0, num_denoising_steps: int = 50, 
    device: str = "cuda",
    save_path: str = "./generated_samples"
) -> torch.Tensor:
    labels = torch.tensor(labels, device=device, dtype=torch.long)
    x_shape = (labels.shape[0], dit.config.in_channels, dit.config.input_size, dit.config.input_size)
    x = torch.randn(x_shape, device=device)

    labels_double = torch.cat([
        labels, 
        torch.full(labels.shape, dit.config.num_classes, device=device)
    ], dim=0)

    t = torch.zeros(2 * x.shape[0], device=x.device)
    dt = 1.0 / num_denoising_steps

    for _ in range(num_denoising_steps):
        x_double = torch.cat([x, x], dim=0)
        pred = dit(x_double, t, labels_double)
        pred_cond, pred_uncond = pred.chunk(2, dim=0)
        pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        x = x + dt * pred
        t += dt

    decoded = ae.decode(x)
    decoded = decoded.clamp(-1.0, 1.0).permute(0, 2, 3, 1)
    
    imgs = (127.5 * (decoded + 1.0)).cpu().numpy()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx, img in enumerate(imgs):
        img = Image.fromarray((127.5 * (img + 1.0)).astype("uint8"))
        img.save(os.path.join(save_path, f"generated_{idx}.png"))


@torch.inference_mode()
def main(labels: list[int], guidance_scale: float, num_denoising_steps: int):
    ae_conf: AutoEncoderParams = ae_configs["ae_256_ch16"]
    ae: AutoEncoder = load_ae(ae_conf, "./checkpoints/ae_ch16.safetensors")
    ae.requires_grad_(False)
    ae.eval()
    ae = ae.to("cuda")

    dic_config: DitConfig = dit_configs["dit_base_256_2_birds"]
    dit: Dit = Dit(dic_config)
    dit.requires_grad_(False)
    dit.eval()
    dit = dit.to("cuda")

    generate(dit, ae, labels, guidance_scale, num_denoising_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images using the DiT model.')
    parser.add_argument('--labels', type=int, nargs='+', required=True,
                        help='List of class labels for image generation')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='Guidance scale for the diffusion process')
    parser.add_argument('--num_denoising_steps', type=int, default=50,
                        help='Number of denoising steps')
    
    args = parser.parse_args()
    
    main(args.labels, args.guidance_scale, args.num_denoising_steps)