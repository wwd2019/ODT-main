# -*- coding: utf-8 -*-

import os
import pickle
from collections import Counter

from tqdm import tqdm

GPU_NUMBER = [2]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
import torch
import datasets
from pathlib import Path
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *

# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
# parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
# parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
# parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
# parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
# parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
# parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
# parser.add_argument("--grad_acc", type=int, default=60, help='Number of gradient accumulation.')
# parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
# parser.add_argument("--mask_prob", type=float, default=0.15, help='Probability of masking.')
# parser.add_argument("--replace_prob", type=float, default=0.9, help='Probability of replacing with [MASK] token for masking.')
# parser.add_argument("--pos_embed", type=bool, default=False, help='Using Gene2vec encoding or not.')
# parser.add_argument("--data_path", type=str, default='/home/wwd/codebox/CompexSystems/scBERT-master/data/spatial_unified_domain_50.h5ad', help='Path of data for pretraining.')
# parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
# parser.add_argument("--model_name", type=str, default='panglao_pretrain', help='Pretrained model name.')
#
# args = parser.parse_args()
local_rank = -1
os.environ["RANK"] ='1'
os.environ['WORLD_SIZE'] = '1' #总进程数
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
rank = int(os.environ["RANK"])
is_master = rank == 0
data_path = './data/spatial_unified_domain_50.h5ad'
SEED = 2021
EPOCHS = 100
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 60
LEARNING_RATE =1e-4
SEQ_LEN =18118
VALIDATE_EVERY = 1
CLASS = 1
MASK_PROB = 0.15
REPLACE_PROB = 0.9
RANDOM_TOKEN_PROB = 0.
MASK_TOKEN_ID = CLASS - 1
PAD_TOKEN_ID = CLASS - 1
MASK_IGNORE_TOKEN_IDS = [0]
POS_EMBED_USING = False

model_name = 'panglao_pretrain'
ckpt_dir = './ckpts/'

# dist.init_process_group(backend='nccl')
device = torch.device("cuda")
# world_size = torch.distributed.get_world_size()
# seed_all(SEED + torch.distributed.get_rank())

# get the random prob matrix and True means smaller than prob threshold
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

# get the mask matrix which cannot be masked
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)      # num of mask of a single sequence in average
    num_tokens = mask.sum(dim=-1, keepdim=True)     # num of pure tokens of each sequence except special tokens
    mask_excess = torch.cat((torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0)))).reshape(mask.size(0),mask.size(-1)).to(device)
    mask_excess = (mask_excess >= (num_tokens * prob).ceil())        # only 15% of pure tokens can be masked
    mask_excess = mask_excess[:, :max_masked]       # get difference between 15% of pure tokens and 15% of all tokens
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)     # rand (0-1) as prob, special token use -1e9
    _, sampled_indices = rand.topk(max_masked, dim=-1)      # get index of topk prob to mask
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)        # delete difference of mask not pure
    new_mask = torch.zeros((batch, seq_len + 1), device=device)     # get (batch, seq_len) shape zero matrix
    new_mask.scatter_(-1, sampled_indices, 1)       # set masks in zero matrix as 1
    return new_mask[:, 1:].bool()       # the final mask, True is mask

def data_mask(data,
    mask_prob = MASK_PROB,
    replace_prob = REPLACE_PROB,
    num_tokens = None,
    random_token_prob = RANDOM_TOKEN_PROB,
    mask_token_id = MASK_TOKEN_ID,
    pad_token_id = PAD_TOKEN_ID,
    mask_ignore_token_ids = MASK_IGNORE_TOKEN_IDS
):
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
    # also do not include these special tokens in the tokens chosen at random
    no_mask = mask_with_tokens(data, mask_ignore_token_ids)   # ignore_token as True, will not be masked later
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)      # get the True/False mask matrix
    # get mask indices
    ## mask_indices = torch.nonzero(mask, as_tuple=True)   # get the index of mask(nonzero value of mask matrix)
    # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
    masked_input = data.clone().detach()
    # if random token probability > 0 for mlm
    if random_token_prob > 0:
        assert num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
        random_token_prob = prob_mask_like(data, random_token_prob)       # get the mask matrix of random token replace
        random_tokens = torch.randint(0, num_tokens, data.shape, device=data.device)     # generate random token matrix with the same shape as input
        random_no_mask = mask_with_tokens(random_tokens, mask_ignore_token_ids)        # not masked matrix for the random token matrix
        random_token_prob &= ~random_no_mask        # get the pure mask matrix of random token replace
        random_indices = torch.nonzero(random_token_prob, as_tuple=True)        # index of random token replace
        masked_input[random_indices] = random_tokens[random_indices]        # replace some tokens by random token
    # [mask] input
    replace_prob = prob_mask_like(data, replace_prob)     # get the mask matrix of token being masked
    masked_input = masked_input.masked_fill(mask * replace_prob, mask_token_id)        # get the data has been masked by mask_token
    # mask out any tokens to padding tokens that were not originally going to be masked
    labels = data.masked_fill(~mask, pad_token_id)        # the label of masked tokens
    return masked_input, labels

class SCDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        return full_seq

    def __len__(self):
        return self.data.shape[0]

adata = sc.read_h5ad(data_path)
adata.obs["celltype"] = adata.obs["celltype"].astype("category")
celltype_mapping = dict(enumerate(adata.obs["celltype"].cat.categories))
# 对细胞种类数值化
adata.obs["celltype"] = adata.obs["celltype"].cat.codes
print(celltype_mapping) #  {0: 'Nb2', 1: 'gut', 2: 'neural', 3: 'epidermal', 4: 'muscle', 5: 'pharynx', 6: 'protonephridia', 7: 'cathepsin_cells', 8: 'parenchymal'}
def p_cell_pretrain(data_path = '/home/wwd/codebox/Geneformer/new/planaria2/data_treated_sets/other_valid_p_cell/p_all_unlabell_cell.dataset'):
    # 使用50万无标签涡虫数据进行预训练
    # with open(data_path, 'rb') as f:
    #     data = pickle.load(f)
    # print(data[0])
    dataset = datasets.Dataset.load_from_disk(data_path)
    print(dataset)
    for
    pass
p_cell_pretrain()
def finetune_timepoint(timepoint,adata):
    # 分时间点训练
    dataset_name =timepoint
    # 分时间点保存
    save_dir = Path('./ckpts/')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    # save the whole script to the dir
    os.system(f"cp {__file__} {save_dir}")
    adata_timepoint = adata[adata.obs["timepoint"] == timepoint] # 对应的数据
    cell_types = adata_timepoint.obs["celltype"]
    num_cell_types = len(Counter(cell_types))
    data = adata_timepoint.X
    # 划分训练集和验证集，同时保持数据和标签对应
    data_train, data_val, cell_types_train, cell_types_val = train_test_split(
        data, cell_types, test_size=0.2, random_state=SEED, stratify=cell_types
    )

    model = PerformerLM(
        num_tokens = CLASS,
        dim = 2048,
        depth = 6,
        max_seq_len = SEQ_LEN,
        heads = 10,
        local_attn_heads = 0,
        g2v_position_emb = POS_EMBED_USING,
        mlp_need= True,
        num_labels=num_cell_types
    )
    model.to('cuda')
    # optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    # learning rate scheduler
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=LEARNING_RATE,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN_ID, reduction='mean').to('cuda')
    softmax = nn.Softmax(dim=-1)
    # 将数据转换为 PyTorch 张量
    data_train_tensor = torch.tensor(data_train.todense(), dtype=torch.float32) if hasattr(data_train,
                                                                                           "todense") else torch.tensor(
        data_train, dtype=torch.float32)
    data_val_tensor = torch.tensor(data_val.todense(), dtype=torch.float32) if hasattr(data_val,
                                                                                       "todense") else torch.tensor(
        data_val, dtype=torch.float32)

    cell_types_train_tensor = torch.tensor(cell_types_train.values, dtype=torch.long) if hasattr(cell_types_train,
                                                                                                 "values") else torch.tensor(
        cell_types_train, dtype=torch.long)
    cell_types_val_tensor = torch.tensor(cell_types_val.values, dtype=torch.long) if hasattr(cell_types_val,
                                                                                             "values") else torch.tensor(
        cell_types_val, dtype=torch.long)
    # 创建 DataLoader
    BATCH_SIZE = 8  # 可调整 batch 大小
    train_loader = DataLoader(list(zip(data_train_tensor, cell_types_train_tensor)), batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(list(zip(data_val_tensor, cell_types_val_tensor)), batch_size=BATCH_SIZE, shuffle=False)
    model.train()
    for i in range(1, EPOCHS+1):
        running_loss = 0.0
        cum_acc = 0.0
        for index, data in tqdm(enumerate(train_loader)):
            index += 1
            data = data.to(device)
            data, labels = data_mask(data)
            logits = model(data)
            loss = loss_fn(logits.transpose(1, 2), labels) / GRADIENT_ACCUMULATION
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()
            final = softmax(logits)[..., 1:-1]
            final = final.argmax(dim=-1) + 1
            pred_num = (labels != PAD_TOKEN_ID).sum(dim=-1)
            correct_num = ((labels != PAD_TOKEN_ID) * (final == labels)).sum(dim=-1)
            cum_acc += torch.true_divide(correct_num, pred_num).mean().item()

        epoch_loss = running_loss / index
        epoch_acc = 100 * cum_acc / index
        if is_master:
            print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
        dist.barrier()
        scheduler.step()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            dist.barrier()
            running_loss = 0.0
            running_error = 0.0
            predictions = []
            truths = []
            with torch.no_grad():
                for index, data in enumerate(val_loader):
                    index += 1
                    data = data.to(device)
                    data, labels = data_mask(data)
                    logits = model(data)
                    loss = loss_fn(logits.transpose(1, 2), labels)
                    running_loss += loss.item()
                    softmax = nn.Softmax(dim=-1)
                    final = softmax(logits)[..., 1:-1]
                    final = final.argmax(dim=-1) + 1
                    predictions.append(final)
                    truths.append(labels)
                del data, labels, logits, final
                # gather
                correct_num = ((truths != PAD_TOKEN_ID) * (predictions == truths)).sum(dim=-1)[0].item()
                val_num = (truths != PAD_TOKEN_ID).sum(dim=-1)[0].item()
                val_loss = running_loss / index

            if is_master:
                val_acc = 100 * correct_num / val_num
                print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | Accuracy: {val_acc:6.4f}%  ==')
        del predictions, truths
        if is_master:
            save_ckpt(i, model, optimizer, scheduler, epoch_loss, model_name, ckpt_dir)

    pass
# finetune_timepoint('0h',adata)



