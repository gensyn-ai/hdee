import logging
import os
import time

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import llama
import sys


def loss_fn(labels, logits, reduction="mean"):
    vocab_size = logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        reduction=reduction,
    )


def get_eval_ppl(model, dev_data_loader, device, eval_steps=-1):
    model.eval()
    dev_loss = torch.tensor(0.0, device=device)
    tokens = torch.tensor(0.0, device=device)
    for step, batch in enumerate(dev_data_loader):
        input_ids = batch["input_ids"].to(device=device)
        labels = batch["label"].to(device=device)
        with torch.no_grad():
            tokens += labels.shape[0] * labels.shape[1]
            dev_loss += loss_fn(labels, model(input_ids), reduction="sum")

        if eval_steps > 0 and step >= eval_steps:
            break

    dist.reduce(dev_loss, dst=0)
    dist.reduce(tokens, dst=0)
    model.train()
    return torch.exp(dev_loss / tokens)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    torch.cuda.set_device(local_rank)

    log_dir = os.path.join(cfg.log_dir, cfg.exp_name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(
        backend=backend,
        device_id=torch.device(device),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
    dist.barrier()
    file_handler = logging.FileHandler(os.path.join(log_dir, f"training_{rank}.log"))
    logger.addHandler(file_handler)

    if rank == 0:
        logger.info(OmegaConf.to_yaml(cfg))
        logger.info(f"Using communication backend: {backend}.")
    dist.barrier()

    tb_root_path = os.path.join(cfg.log_dir, "tensorboard")
    tb_log_path = os.path.join(tb_root_path, cfg.exp_name)
    
    if rank == 0:
        logger.info("Creating folders ...")
        if not os.path.isdir(tb_root_path):
            os.mkdir(tb_root_path)
        if not os.path.isdir(tb_log_path):
            os.mkdir(tb_log_path)
        

    writer = SummaryWriter(log_dir=tb_log_path) if rank == 0 else None
    
    validation_data_loaders = {
        key: torch.utils.data.DataLoader(
            instantiate(value), batch_size=cfg.batch_size, num_workers=1
        )
        for key, value in cfg.validation_data_loaders.items()
    }
    
    model = torch.load(cfg.model.model_dir, weights_only=False)
    model.eval()

    eval_steps = cfg.eval_steps

    dev_losses = {
        key: get_eval_ppl(model, validation_data_loader, device, eval_steps)
        for key, validation_data_loader in validation_data_loaders.items()
    }

    logger.info(
        "Loss: ".join(
            [
                f"{key}_ppl: {(dev_loss.item()):.1f}, "
                for key, dev_loss in dev_losses.items()
            ]
        )
    )


    torch.distributed.destroy_process_group()
    logging.info("Exited successfully.")
    file_handler.flush()


if __name__ == "__main__":
    main()
