import os
import tqdm

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from omegaconf import OmegaConf
from safetensors.torch import save_file

from src.dit import Dit
from src.config import DitConfig, dit_configs, ae_configs
from src.autoencoder import AutoEncoder, AutoEncoderParams
from src.utils import load_dit, load_ae

from image_datasets.datasets import BirdImageDataset
from inference import generate as generate_images


def train(rank, world_size, dit, ae, dataloader, optimizer, training_config, checkpointing_config, device):
    torch.cuda.set_device(rank)
    dit = DDP(dit, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    data_iter = iter(dataloader)
    checkpoint_every = checkpointing_config.checkpoint_every
    generate_every = checkpointing_config.generate_every

    bar = tqdm.tqdm(total=training_config.num_steps, position=rank, disable=(rank != 0))
    loss_history = []

    for global_step in range(training_config.num_steps * training_config.gradient_accumulation_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images, labels = batch
        images = images.to(device)
        labels = torch.LongTensor(labels).to(device)

        with torch.no_grad():
            images = 2.0 * images - 1.0
            x_1 = ae.encode(images.to(torch.float32))

        x_0 = torch.randn_like(x_1)
        t = torch.sigmoid(torch.randn((images.shape[0],), device=device))
        tr = t.view(-1, 1, 1, 1)
        x_t = tr * x_1 + (1 - tr) * x_0

        velocity = dit(x_t, t, labels)
        loss = torch.nn.functional.mse_loss(velocity, x_1 - x_0)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        bar.update(1)
        bar.set_description(f"Loss: {loss.item()}")

        if rank == 0:
            loss_history.append(loss.item())
        
        if rank == 0 and global_step % checkpoint_every == 0:
            state_dict = dit.module.state_dict()
            save_file({k: v.contiguous() for k, v in state_dict.items()}, os.path.join(checkpointing_config.save_dir, f"dit_{global_step}.safetensors"))

        if rank == 0 and global_step % generate_every == 0:
            with torch.no_grad():
                dit.eval()
                images_save_path = os.path.join(checkpointing_config.save_dir, f"../samples/{global_step}")
                os.makedirs(images_save_path, exist_ok=True)
                generate_images(dit.module, ae, labels.detach().cpu().tolist(), device=device, save_path=images_save_path)
                dit.train()

    if rank == 0:
        torch.save(loss_history, "loss_history.pth")
    
    bar.close()

def main():
    dist.init_process_group(backend='nccl')
    
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    
    args = OmegaConf.load("training_configs/config.yaml")

    ae_conf: AutoEncoderParams = ae_configs["ae_256_ch16"]
    ae_checkpoint_path = "/workspace/checkpoints/ae_ch16.safetensors"
    ae: AutoEncoder = load_ae(ae_conf, ae_checkpoint_path, device=device)
    ae.requires_grad_(False)
    ae.eval()

    dic_config: DitConfig = dit_configs["dit_base_256_2_birds"]
    dit: Dit = load_dit(dic_config, None, device=device)
    dit.train()

    print(sum(p.numel() for p in dit.parameters() if not p.requires_grad))

    dit = dit.to(device)
    optimizer = torch.optim.AdamW(dit.parameters(), **args.optim)

    dataset = BirdImageDataset(**args.dataset)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=args.training.batch_size, sampler=sampler, **args.dataloader)
    
    train(local_rank, world_size, dit, ae, dataloader, optimizer, args.training, args.checkpointing, device)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()