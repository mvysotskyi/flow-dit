import os

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from safetensors.torch import save as sft_save

from image_datasets.datasets import BirdImageDataset

from src.dit import Dit
from src.config import DitConfig, dit_configs, ae_configs
from src.autoencoder import AutoEncoder, AutoEncoderParams
from src.utils import load_dit, load_ae

from inference import generate as generate_images


def logisticnormal_like(shape: list[int], device):
    return torch.distributions.LogisticNormal(
        torch.zeros(shape, device=device),
        torch.ones(shape, device=device)
    )

def train(dit: Dit, ae: AutoEncoder, dataloader, optimizer, training_config, checkpointing_config, device):
    data_iter = iter(dataloader)
    checkpoint_every = checkpointing_config.checkpoint_every
    generate_every = training_config.generate_every

    global_step = 0

    for step in range(training_config.num_steps * training_config.gradient_accumulation_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images, labels = batch
        images = images.to(device)

        with torch.no_grad():
            images = images.permute(0, 3, 1, 2)
            x_1 = ae.encode(images.to(torch.float32))

        labels = torch.LongTensor(labels).to(device)

        x_0 = torch.randn_like(x_1)
        t = logisticnormal_like(images.shape[0], device)
        x_t = (1 - t) * x_1 + t * x_0

        if step % training_config.gradient_accumulation_steps == 0:
            velocity = dit(x_t, t, labels)
            loss = torch.functional.F.mse_loss(velocity, x_1 - x_0)
            loss /= training_config.gradient_accumulation_steps
            loss.backward()

            global_step += 1
            if global_step % checkpoint_every == 0:
                sft_save(dit.state_dict(), os.path.join(checkpointing_config.save_dir, f"dit_{global_step}.safetensors"))

            if global_step % generate_every:
                with torch.no_grad():
                    dit.eval()
                    images_save_path = os.path.join(checkpointing_config.save_dir, f"./samples/{global_step}")
                    os.makedirs(images_save_path, exist_ok=True)
                    generate_images(dit, ae, labels, device=device, save_path=images_save_path)
                    dit.train()

            optimizer.step()
            optimizer.zero_grad()
        else:
            velocity = dit(x_t, t, labels)
            loss = torch.functional.F.mse_loss(velocity, x_1 - x_0)
            loss /= float(training_config.gradient_accumulation_steps)
            loss.backward()



if __name__ == "__main__":
    args = OmegaConf.load("training_configs/config.yaml")

    ae_conf: AutoEncoderParams = ae_configs["ae_256_ch16"]
    ae_checkpoint_path = "./checkpoints/ae_ch16.safetensors"
    ae: AutoEncoder = load_ae(ae_conf, ae_checkpoint_path, device="cuda")
    ae.requires_grad_(False)
    ae.eval()

    dic_config: DitConfig = dit_configs["dit_base_256_2_birds"]
    dit: Dit = load_dit(dic_config, None, device="cuda")
    dit.train()
    dit.requires_grad_(True)

    optimizer = torch.optim.AdamW(dit.parameters(), **args.optim)

    dataset = BirdImageDataset(**args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.training.batch_size, **args.dataloader)

    train(dit, ae, dataloader, optimizer, args.training, args.checkpointing, "cuda")

    # Save final model
    sft_save(dit.state_dict(), os.path.join(args.checkpointing.save_dir, f"dit_final.safetensors"))

