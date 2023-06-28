# package import
import os
from typing import Type
import torch
import torch.nn.functional as F
import torchvision
import pandas as pd
from torch.utils.data.dataloader import DataLoader
# import wandb
import utils_builder
import math
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# image-text embedding diagnosis style trainer Class (with language model)


class trainer_wBert:
    def __init__(self, model,
                 optimizer, device, model_name, **args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.train_batch_size = args['batch_size']
        self.test_batch_size = args['test_batch_size']
        self.max_epochs = args['max_epochs']
        self.lr_max = args['lr']
        self.num_workers = args['num_workers']
        self.checkpoint_interval = args['checkpoint_interval']

    def orthogonal_loss(self, x1, x2):
        def off_diagonal(x):
            # return a flattened view of the off-diagonal elements of a square matrix
            n, m = x.shape
            assert n == m
            return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        logits = torch.mm(x1.T, x2).to(self.device)

        logits.div_(self.train_batch_size)
        on_diag = torch.diagonal(logits).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(logits).pow_(2).sum()
        loss = on_diag + 0.0051*off_diag
        return loss/2

    def align_loss(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        loss = 2 - 2 * (x * y).sum(dim=-1)
        loss += 2 - 2 * (y * x).sum(dim=-1)
        return loss.mean()



    # traing process
    def train_w_TextEmb(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size,
                                  num_workers=self.num_workers,
                                  drop_last=True, shuffle=False,
                                  sampler=DistributedSampler(train_dataset))

        model_checkpoints_folder = os.path.join('../checkpoints')
        if not os.path.exists(model_checkpoints_folder):
            print('create directory "{}" for save checkpoint!'.format(
                model_checkpoints_folder))
            print('---------------------------')
            os.mkdir(model_checkpoints_folder)
        else:
            print('directory "{}" existing for save checkpoint!'.format(
                model_checkpoints_folder))

        # automatically resume from checkpoint if it exists
        print('#########################################')
        print('Be patient..., checking checkpoint now...')
        if os.path.exists(model_checkpoints_folder + self.model_name+'_checkpoint.pth'):
            ckpt = torch.load(model_checkpoints_folder + self.model_name+'_checkpoint.pth',
                              map_location='cpu')
            start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('continue training successful!')
        else:
            start_epoch = 0
            print('Start training from 0 epoch')

        print('#########################################')
        print('training start!')

        # scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=int(self.max_epochs*len(train_dataset) //
                    4//self.train_batch_size * 0.4),
            T_mult=1,
            eta_min=1e-8,
        )
        niter = 1

        skip_scheduler = False
        scaler = GradScaler()

        for epoch_counter in tqdm(range(start_epoch, self.max_epochs+1)):

            epoch_loss = 0
            epoch_loss_orthogonal, epoch_loss_align = 0, 0

            for data in tqdm(train_loader):
                # get raw text
                imp = data['raw_text']

                # get image
                img = data['image'].to(torch.float32).to(
                    self.device).contiguous()

                self.optimizer.zero_grad()

                # amp style (might decrease precision)
                with autocast():
                    imp_tokenize_output = self.model.module._tokenize(imp)

                    input_ids = imp_tokenize_output.input_ids.to(
                        self.device).contiguous()
                    attention_mask = imp_tokenize_output.attention_mask.to(
                        self.device).contiguous()

                    output_dict = self.model(img, input_ids, attention_mask) 
                    img_emb, proj_img_emb, proj_text_emb = output_dict['img_emb'], output_dict['proj_img_emb'], output_dict['proj_text_emb']

                    loss_orthogonoal = self.orthogonal_loss(img_emb, img_emb)
                    loss_align = self.align_loss(proj_img_emb, proj_text_emb)

                    loss = loss_orthogonoal + loss_align
                    # accumalate loss for logging
                    epoch_loss += loss.item()
                    epoch_loss_orthogonal += loss_orthogonoal.item()
                    epoch_loss_align += loss_align.item()
                    
                    # if self.device == 0:
                    #     print(
                    #         f'epoch {epoch_counter} iter {niter} loss is {loss.item()},\
                    #         orthogonal loss is {loss_orthogonoal.item()},\
                    #         align loss is {loss_align.item()}')


                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()

                    if not skip_scheduler:
                        scheduler.step()
                niter += 1

            if self.device == 0:
                
                epoch_iter = (len(train_dataset)//self.train_batch_size//4)
                print(f'{epoch_counter} epoch loss is {epoch_loss/epoch_iter}!')

                if epoch_counter % 10 == 0:
                    torch.save(self.model.module.state_dict(),
                               model_checkpoints_folder + self.model_name+f'_{epoch_counter}'+'_total.pth')

        # save final vision encoder
        torch.save(self.model.module.encoder.state_dict(),
                   model_checkpoints_folder + self.model_name+'_encoder.pth')
        # save final total model
        torch.save(self.model.module.state_dict(),
                   model_checkpoints_folder + self.model_name+'_total.pth')

    def save_checkpoints(self, epoch, PATH):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            PATH)
