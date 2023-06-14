import random
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
import tempfile
import os
from torch import optim
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torchvision
import torch
from torch.utils.data.dataloader import DataLoader
import yaml
import sys
sys.path.append("../utils")
from utils_trainer import trainer_wBert
import utils_dataset
import utils_builder

# import wandb


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def ddp_main():
    dist.init_process_group("nccl")
    torch.cuda.empty_cache()
    rank = dist.get_rank()

    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()

    # set up
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    # loading data path
    text_path = config['text_path']
    img_path = config['img_path']

    # define image-text dataset
    train_dataset = utils_dataset.I_T_emb_dataset(
        image_path=img_path, csv_path=text_path)
    train_dataset = train_dataset.get_dataset(train_test='train')

    # building model part
    # --------------------
    if config['network']['img_model'] == 'resnet50':
        model = utils_builder.ResNet_CXRBert()

    '''
    you can freeze bert from last layer to first layer.
    set num of layer in config.yaml
    default is freeze 9 layers
    '''
    if config['network']['free_layers'] is not None:
        for layer_idx in range(int(config['network']['free_layers'])):
            for param in list(model.lm_model.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False

    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    # --------------------

    # choose optimizer (no LARS, AdamW with small batch)
    # --------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **config['optimizer']['params'],
        betas=(0.9, 0.999)
    )

    # ---------xw-----------
    trainer = trainer_wBert(model=model,
                            optimizer=optimizer,
                            device=rank,
                            model_name=config['wandb_name'],
                            **config['trainer'])
    # --------------------
    
    # --------------------
    # I_T_P_trainer
    trainer.train_w_TextEmb(train_dataset)


ddp_main()
