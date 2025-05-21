# -*- coding: utf-8 -*-

import os
# GPU_NUMBER = [7,8,9]
# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
import pickle
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


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
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
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

torch.cuda.empty_cache()
torch.cuda.set_device(8)
device = torch.device('cuda',8)
# device = torch.device('cpu')
# local_rank = -1
os.environ["RANK"] ='1'
os.environ['WORLD_SIZE'] = '1' #总进程数
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
rank = int(os.environ["RANK"])
is_master = rank == 0

SEED = 2021
EPOCHS = 40
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 10
LEARNING_RATE =1e-4
# SEQ_LEN =22195
SEQ_LEN = 4622
VALIDATE_EVERY = 1
CLASS = 2
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
        # print(self.data[rand_start])
        # print(type(self.data[rand_start]))
        # full_seq = np.array([full_seq])
        full_seq = self.data[rand_start]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to('cuda')
        return full_seq

    def __len__(self):
        return self.data.shape[0]
def p_cell_pretrain(data_path = '/home/wwd/codebox/Geneformer/new/planaria2/data_treated_sets/other_valid_p_cell/unlabelled_cells_over50.h5ad'):
    # 使用50万无标签涡虫数据进行预训练
    save_dir = Path('./ckpts/')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    adata = sc.read_h5ad(data_path)
    data = adata.X

    data_train, data_val = train_test_split(
        data, test_size=0.05, random_state=SEED
    )
    model = PerformerLM(
        num_tokens = CLASS,
        dim = 200,
        depth = 6,
        max_seq_len = SEQ_LEN,
        heads = 10,
        local_attn_heads = 0,
        g2v_position_emb = POS_EMBED_USING
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
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction='mean').to('cuda')
    train_dataset = SCDataset(data_train)
    val_dataset = SCDataset(data_val)

    # 创建 DataLoader
    BATCH_SIZE = 4  # 可调整 batch 大小
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.train()
    for i in tqdm(range(1, EPOCHS + 1)):
        running_loss = 0.0
        cum_acc = 0.0
        for index, data in enumerate(train_loader):
            index += 1
            if(index % 100 == 0):
                print(f'Epoch: {i}/{EPOCHS + 1} ({index/len(train_loader):.5f}')
            data = data.to(device)
            data, labels = data_mask(data)
            logits = model(data)

            loss = loss_fn(logits.transpose(1, 2), labels) / GRADIENT_ACCUMULATION
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e2))
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss.item()


        epoch_loss = running_loss / index
        epoch_acc = 100 * cum_acc / index

        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
        scheduler.step()
        if i % VALIDATE_EVERY == 0:
            model.eval()

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
                    #print(logits)
                    softmax = nn.Softmax(dim=-1)
                    final = softmax(logits)

                    final = final.argmax(dim=-1)

                    predictions.append(final)
                    truths.append(labels)
                del data, labels, logits, final
                # gather
                correct_num = ((truths != PAD_TOKEN_ID) * (predictions == truths)).sum(dim=-1)[0].item()
                val_num = (truths != PAD_TOKEN_ID).sum(dim=-1)[0].item()
                val_loss = running_loss / index
            #if is_master:
            val_acc = 100 * correct_num / val_num
            print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | Accuracy: {val_acc:6.4f}%  ==')
        del predictions, truths
        if(i%20 ==0 ): # 每20epoch保存
            torch.save(model,f"./ckpts/pretrained_epoch_{i}_model.pth")


    pass
# p_cell_pretrain()

class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to('cuda')
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]


class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def finetune_timepoint(timepoint,adata,celltype_mapping):
    UNASSIGN_THRES = 0.0
    PATIENCE = 3 # 3次没有增长，就停止
    # 分时间点训练
    dataset_name =timepoint
    # 分时间点保存
    save_dir = Path('./ckpts/'+timepoint+'/')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    # save the whole script to the dir
    os.system(f"cp {__file__} {save_dir}")
    adata_timepoint = adata[adata.obs["timepoint"] == timepoint] # 对应的数据
    cell_types = adata_timepoint.obs["celltype"].tolist()
    data = adata_timepoint.X
    # 创建 DataLoader
    BATCH_SIZE = 1  # 可调整 batch 大小
    label = torch.Tensor(cell_types)
    acc_list = []
    f1_list = []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    for index_train, index_val in sss.split(data, label):
        data_train, label_train = data[index_train], label[index_train]
        data_val, label_val = data[index_val], label[index_val]
        train_dataset = SCDataset(data_train, label_train)
        val_dataset = SCDataset(data_val, label_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    model = PerformerLM(
        num_tokens=CLASS,
        dim=200,
        depth=6,
        max_seq_len=SEQ_LEN,
        heads=10,
        local_attn_heads=0,
        g2v_position_emb=POS_EMBED_USING
    )
    model.to_out = Identity(dropout=0., h_dim=128, out_dim=len(celltype_mapping))
    model = model.to(device)

    # optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=LEARNING_RATE,
        min_lr=1e-5,
        warmup_steps=5,
        gamma=0.9
    )
    loss_fn = nn.CrossEntropyLoss(weight=None).to(device)
    scaler = GradScaler()
    max_acc = 0
    trigger_times = 0
    acc_list = []
    f1_list = []

    for i in tqdm(range(1, EPOCHS + 1)):
        model.train()
        running_loss = 0.0
        cum_acc = 0.0

        for data, labels in tqdm(train_loader, total=len(train_loader)):
            data, labels = data.to(device), labels.to(device)

            # 混合精度训练
            with autocast():
                logits, _ = model(data, output_attentions=True)
                labels = labels.long()
                loss = loss_fn(logits, labels)

            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            optimizer.zero_grad()  # 清空梯度
            scaler.update()

            running_loss += loss.item()

            # 计算准确率
            with torch.no_grad():
                final = logits.argmax(dim=-1)
                correct_num = (final == labels).sum().item()
                pred_num = labels.size(0)
                cum_acc += correct_num / pred_num

            torch.cuda.empty_cache()

        print(f"Epoch {i} Loss: {running_loss:.4f}, Accuracy: {cum_acc / len(train_loader):.4f}")
        scheduler.step()

        # 评估阶段
        model.eval()
        cum_correct = 0
        cum_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data_v, labels_v in tqdm(val_loader, total=len(val_loader), dynamic_ncols=True, leave=True, position=0,
                                         disable=False):
                data_v, labels_v = data_v.to(device), labels_v.to(device)

                logits, _ = model(data_v, output_attentions=True)
                final = logits.argmax(dim=-1)

                # 更新准确率统计
                cum_correct += (final == labels_v).sum().item()
                cum_total += labels_v.size(0)

                # 收集 F1 计算数据
                all_preds.extend(final.cpu().tolist())
                all_labels.extend(labels_v.cpu().tolist())

        # 计算整体验证集准确率
        cur_acc = cum_correct / cum_total

        # 计算全局 F1 Score
        avg_f1 = f1_score(all_labels, all_preds, average='macro')
        acc_list.append(cur_acc)
        f1_list.append(avg_f1)
        print(f'    ==  Epoch: {i} | Validation Acc: {cur_acc:.6f} | F1 Score: {avg_f1:.6f}  ==')

        # 记录最好的模型
        if cur_acc > max_acc:
            max_acc = cur_acc
            trigger_times = 0
            os.makedirs(f'./ckpts/{timepoint}', exist_ok=True)
            torch.save(model.state_dict(), f'./ckpts/{timepoint}/best_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= PATIENCE:
                break

        torch.cuda.empty_cache()
    with open('./ckpts/'+timepoint+'/'+'history.txt','w')as file:
        for id in range(len(acc_list)):
            file.write(str(acc_list[id])+' ' + str(f1_list[id]))
            file.write('\n')
    pass
data_path = './data/spatial_unified_domain_50.h5ad'
adata = sc.read_h5ad(data_path)
sc.pp.filter_genes(adata, min_cells=2000)
print(adata)
adata.obs["celltype"] = adata.obs["celltype"].astype("category")
celltype_mapping = dict(enumerate(adata.obs["celltype"].cat.categories))
# 对细胞种类数值化
adata.obs["celltype"] = adata.obs["celltype"].cat.codes
print(celltype_mapping) #  {0: 'Nb2', 1: 'gut', 2: 'neural', 3: 'epidermal', 4: 'muscle', 5: 'pharynx', 6: 'protonephridia', 7: 'cathepsin_cells', 8: 'parenchymal'}
batch_id_mapping = {batch: idx for idx, batch in enumerate(adata.obs['timepoint'].unique())}
for timepoint in batch_id_mapping.keys():
    if(timepoint != 'WT'):
        print(timepoint + ' pass')
        continue
    finetune_timepoint(timepoint,adata,celltype_mapping)



