import logging
import os
import time
import numpy as np

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
import sys
from pathlib import Path
from btm_utils import *

def loss_fn(labels, logits, reduction="mean"):
    vocab_size = logits.shape[-1]
    return torch.nn.functional.nll_loss(
        logits.view(-1, vocab_size),
        labels.view(-1),
        reduction=reduction,
    )

def precompute_domain_prior(cfg, ensemble_models, dev_data_loader, device, max_sequences=100, decay=0.3):
    
    #Iterate over batches until desired number of sequence has been accounted for
    sequences_considered = 0
    for step, batch in enumerate(dev_data_loader):
        input_ids = batch["input_ids"].to(device=device)
        labels = batch["label"].to(device=device)
        log_probs = {}
        for model_name, model_config in cfg.models.items():
            with torch.no_grad():
                logits= ensemble_models[model_name](input_ids).to(device=device)
            #Convert logits to log_probs per token per batch
            log_probs[model_name] = logits_to_log_probs(logits, labels)
        #Get log likelihood across sequences and experts
        model_to_idx = {expert:idx for idx,expert in enumerate(log_probs)}
        log_prior = torch.tensor([0.0 for expert in log_probs])
        log_probs = torch.stack([log_probs[expert] for expert in log_probs], dim=0)
        for cum_log_prob_seq in torch.cumsum(log_probs, dim=2).unbind(dim=1):
            cur_decay = np.power(decay,sequences_considered+1)
            log_likelihood = cum_log_prob_seq[:,-1] + log_prior
            log_posterior = torch.nn.functional.log_softmax(log_likelihood, dim=0)
            m = log_prior.max()
            log_prior = (
                torch.log(cur_decay * torch.exp(log_prior - m) + torch.exp(log_posterior - m)) + m
            )
            sequences_considered += 1
            if sequences_considered >= max_sequences:
                return {expert:torch.nn.functional.softmax(log_prior, dim=0)[model_to_idx[expert]] for expert in model_to_idx}
    #If we run through all sequences for some reason, just return the latest prior we found
    return {expert:torch.nn.functional.softmax(log_prior, dim=0)[model_to_idx[expert]] for expert in model_to_idx}

def get_ensemble_eval_ppl(cfg, ensemble_models, dev_data_loader, device, eval_steps=-1, domain_distribution='cached_prior', prior=None):
    
    dev_loss = torch.tensor(0.0, device=device)
    tokens = torch.tensor(0.0, device=device)

    for step, batch in enumerate(dev_data_loader):
        input_ids = batch["input_ids"].to(device=device)
        labels = batch["label"].to(device=device)
        probs, log_probs = {}, {}
        for model_name, _ in cfg.models.items():
            with torch.no_grad():
                logits= ensemble_models[model_name](input_ids).to(device=device)
            #Directly get probs from logits
            if domain_distribution == 'simple_avg':
                probs[model_name]= torch.nn.functional.log_softmax(logits, dim=-1)
            #Convert logits to log_probs per token per batch
            elif domain_distribution in ['uniform_prior', 'cached_prior']:
                log_probs[model_name] = logits_to_log_probs(logits, labels)
        #Run through models, accumulate probabilities, normalize uniformly across experts
        if domain_distribution == 'simple_avg':
            weighted_probs=torch.zeros(labels.shape[0],labels.shape[1],128002).to(device=device)
            for model_name in probs:
                weighted_probs+=probs[model_name]
            weighted_probs/=len(probs)
            if dist.get_rank()== 0:
                with torch.no_grad():
                    tokens += labels.shape[0] * labels.shape[1]
                    dev_loss += loss_fn(labels, weighted_probs, reduction="sum")
        #Compute this domain's posterior and accumulate probabilities weighted by prior 
        elif domain_distribution in ['uniform_prior', 'cached_prior']:
            #Move log_probs and log_prior into properly shaped tensors
            log_prior = torch.tensor([torch.log(prior[expert]) for expert in log_probs])
            log_probs = torch.stack([log_probs[expert] for expert in log_probs], dim=0)
            #Get likelihood per model per sequence
            num_experts, batch_size, _ = log_probs.shape
            log_probs_ = torch.cat(
                [torch.zeros(num_experts, batch_size, 1), log_probs[..., :-1]], dim=2
                )
            log_cum_prob = torch.cumsum(log_probs_, dim=2)
            #Compute posterior
            posterior_prob = torch.nn.functional.softmax(
                log_cum_prob + log_prior.unsqueeze(dim=1).repeat(1,batch_size).unsqueeze(dim=2), dim=0
            )
            #Weight ELM probs per token by corresponding posterior
            m = torch.amax(log_probs, dim=(0,1), keepdim=True)
            log_elm = torch.log((posterior_prob * torch.exp(log_probs - m)).sum(dim=0)) + m.squeeze(dim=0)
            if dist.get_rank()== 0:
                with torch.no_grad():
                    tokens += labels.shape[0] * labels.shape[1]
                    dev_loss -= log_elm.sum()
        else:
            raise Exception("The specified domain distribution is not implemented or mispelled.") 

        if eval_steps > 0 and step >= eval_steps:
            break

    return torch.exp(dev_loss / tokens)


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    torch.cuda.set_device(local_rank)

    loc_dir = Path(__file__).parents[3]
    log_dir = os.path.join(loc_dir, cfg.log_dir, cfg.exp_name)

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

    tb_root_path = os.path.join(loc_dir, cfg.log_dir, "tensorboard")
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
        instantiate({'_target_': value._target_, 'path': os.path.join(loc_dir,value.path)}), batch_size=cfg.batch_size, num_workers=0
        )
        for key, value in cfg.validation_data_loaders.items()
    }
    
    #Load trained ELMs and get mapping from model to domain
    ensemble_models, model_to_domain_mapping, domain_to_model_mapping = {}, {}, {}
    for model_name, model_config in cfg.models.items():
        model_dir = os.path.join(loc_dir, model_config.model_dir)
        ensemble_models[model_name] = torch.load(model_dir, weights_only=False)
        ensemble_models[model_name].eval()
        #TODO: Can remove these dicts since not currently used, but leaving in case we want these for logging/visualization purposes.
        try:
            dom = model_config.model_domain
        except:
            dom = model_config['model_dir'].split('/')[-3].split('_domain_')[-1]
        model_to_domain_mapping[model_name] = dom
        domain_to_model_mapping[dom] = model_name

    # We test all cases
    for domain_distribution in ['simple_avg', 'uniform_prior','cached_prior']:
        domain_prior = {}
        for domain, validation_data_loader in validation_data_loaders.items():
            if domain_distribution == 'cached_prior':
                domain_prior[domain] = precompute_domain_prior(cfg, ensemble_models, validation_data_loader, device, max_sequences=100, decay=0.3)
            elif domain_distribution == 'uniform_prior':
                domain_prior[domain] = {expert:torch.tensor(1.0, device=device) for expert in ensemble_models}
            elif domain_distribution == 'simple_avg':
                domain_prior[domain] = None #TODO: Redundant if same weighting mechanism used for all data domains, but left this in case we decide to try different weighting mechanisms per domain/domain "type"

        eval_steps = cfg.eval_steps

        dev_losses = {
            key: get_ensemble_eval_ppl(cfg, ensemble_models, validation_data_loader, device, eval_steps, domain_distribution, domain_prior[key])
            for key, validation_data_loader in validation_data_loaders.items()
        }

        logger.info(
            f"\n---Ensemble method: {domain_distribution}--- "+ 
            " ".join(
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
