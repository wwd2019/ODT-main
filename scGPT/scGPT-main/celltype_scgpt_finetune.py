import os
GPU_NUMBER = [6]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
import pickle
from collections import Counter
from pathlib import Path
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import torch
from anndata import AnnData
import scanpy as sc
import scvi
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.trainer import SeqDataset, predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.append("../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler, test, evaluate
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"

hyperparameter_defaults = dict(
    seed=42,
    dataset_name="p_cell_unlabelled",
    do_train=True,
    # load_model="./scGPT_model",
    load_model=None,
    mask_ratio=0.4,
    epochs=40,
    n_bins=51,
    GEPC=True,  # Masked value prediction for cell embedding
    ecs_thres=0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=1.0,
    lr=1e-4,
    batch_size=8,
    layer_size=128,
    nlayers=4,
    nhead=4,
    CLS = True,
    task= "annotation",
    # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=0.2,
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    log_interval=100,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    device="cuda:0"
)
from types import SimpleNamespace
# 基本参数
config = SimpleNamespace(**hyperparameter_defaults)
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = hyperparameter_defaults['mask_ratio']
mask_value = -1
pad_value = -2
n_input_bins = hyperparameter_defaults['n_bins']

# n_hvg = 20000  # number of highly variable genes
n_hvg = 2048
max_seq_len = n_hvg + 1
per_seq_batch_sample = True
DSBN = True  # Domain-spec batchnorm
explicit_zero_prob = True  # whether explicit bernoulli for zeros
adata = sc.read_h5ad('./data/spatial_unified_domain_50.h5ad')
adata.obs["celltype"] = adata.obs["celltype"].astype("category")
celltype_mapping = dict(enumerate(adata.obs["celltype"].cat.categories))
adata.obs["celltype"] = adata.obs["celltype"].cat.codes
print(celltype_mapping) #  {0: 'Nb2', 1: 'gut', 2: 'neural', 3: 'epidermal', 4: 'muscle', 5: 'pharynx', 6: 'protonephridia', 7: 'cathepsin_cells', 8: 'parenchymal'}
batch_id_mapping = {batch: idx for idx, batch in enumerate(adata.obs['timepoint'].unique())}
adata.obs['batch_id'] = 0
print(batch_id_mapping)




def model_train_timepoint(timepoint, adata):
    # 分时间点训练
    dataset_name =timepoint
    # 分时间点保存
    save_dir = Path(f"./save/dev_{dataset_name}2048/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    # save the whole script to the dir
    os.system(f"cp {__file__} {save_dir}")
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    adata_timepoint = adata[adata.obs["timepoint"] == timepoint] # 对应的数据
    adata = adata_timepoint
    # 准备构建模型并初始化
    embsize = hyperparameter_defaults['layer_size']
    nhead = hyperparameter_defaults['nhead']
    nlayers = hyperparameter_defaults['nlayers']
    d_hid = hyperparameter_defaults['layer_size']
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=3,  # step 1
        filter_cell_by_counts=True,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=True,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3",
        binning=hyperparameter_defaults['n_bins'],  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key=None)  # 单个batch_id数据量不多，不用分批
    input_layer_key = "X_binned"
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["smesg"].tolist()
    celltypes_labels = adata.obs["celltype"].tolist()  # make sure count from 0
    num_types = len(set(celltypes_labels))
    celltypes_labels = np.array(celltypes_labels)
    print(Counter(celltypes_labels))
    batch_ids = adata.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)
    (
        train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,
    ) = train_test_split(
        all_counts, celltypes_labels, batch_ids, test_size=0.2, shuffle=True
    ) # 80%训练20%测试

    # 新创建词汇表
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)  # size of vocabulary
    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=True,
    )

    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    def prepare_dataloader(
            data_pt: Dict[str, torch.Tensor],
            batch_size: int,
            shuffle: bool = False,
            intra_domain_shuffle: bool = False,
            drop_last: bool = False,
            num_workers: int = 0,
    ) -> DataLoader:
        dataset = SeqDataset(data_pt)
        if per_seq_batch_sample:
            # find the indices of samples in each seq batch
            subsets = []
            batch_labels_array = data_pt["batch_labels"].numpy()
            for batch_label in np.unique(batch_labels_array):
                batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
                subsets.append(batch_indices)
            data_loader = DataLoader(
                dataset=dataset,
                batch_sampler=SubsetsBatchSampler(
                    subsets,
                    batch_size,
                    intra_subset_shuffle=intra_domain_shuffle,
                    inter_subset_shuffle=shuffle,
                    drop_last=drop_last,
                ),
                num_workers=num_workers,
                pin_memory=True,
            )
            return data_loader

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
        masked_values_train = random_mask_value(
            tokenized_train["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        masked_values_valid = random_mask_value(
            tokenized_valid["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        print(
            f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
            f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
        )

        input_gene_ids_train, input_gene_ids_valid = (
            tokenized_train["genes"],
            tokenized_valid["genes"],
        )

        input_type_ids_train, input_type_ids_valid = (
            train_celltype_labels,
            valid_celltype_labels,
        )

        input_values_train, input_values_valid = masked_values_train, masked_values_valid
        target_values_train, target_values_valid = (
            tokenized_train["values"],
            tokenized_valid["values"],
        )
        # print(Counter(train_batch_labels))
        tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
        tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

        # if sort_seq_batch:
        #     train_sort_ids = np.argsort(train_batch_labels)
        #     input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        #     input_values_train = input_values_train[train_sort_ids]
        #     input_type_ids_train = input_gene_ids_train[train_sort_ids]
        #     target_values_train = target_values_train[train_sort_ids]
        #     tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        #     valid_sort_ids = np.argsort(valid_batch_labels)
        #     input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        #     input_values_valid = input_values_valid[valid_sort_ids]
        #     input_type_ids_valid = input_type_ids_valid[valid_sort_ids]
        #     target_values_valid = target_values_valid[valid_sort_ids]
        #     tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        train_data_pt = {
            "gene_ids": input_gene_ids_train,
            "celltypes": input_type_ids_train,
            "values": input_values_train,
            "target_values": target_values_train,
            "batch_labels": tensor_batch_labels_train,
        }
        # print(len(input_type_ids_valid))
        # print(input_type_ids_valid[0])

        valid_data_pt = {
            "gene_ids": input_gene_ids_valid,
            "celltypes":input_type_ids_valid,
            "values": input_values_valid,
            "target_values": target_values_valid,
            "batch_labels": tensor_batch_labels_valid,
        }

        return train_data_pt, valid_data_pt

    def test(model: nn.Module, loader: DataLoader) -> None:
        device = config.device
        model.eval()
        num_batches = len(loader)
        predictions = []
        labels_all = []
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltypes"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None,
                    MVC=hyperparameter_defaults['GEPC'],
                    ECS=hyperparameter_defaults['ecs_thres'] > 0,
                    CLS=config.CLS
                )
            masked_positions = input_values.eq(mask_value)  # the postions to predict
            labels_all += celltype_labels
            _, predicted = torch.max(output_dict["cls_output"], 1)
            predicted = predicted.tolist()
            predictions += predicted
        predictions = torch.Tensor(predictions)
        labels_all = torch.Tensor(labels_all)
        # compute accuracy, precision, recall, f1
        accuracy = accuracy_score(labels_all, predictions)
        precision = precision_score(labels_all, predictions, average="macro")
        recall = recall_score(labels_all, predictions, average="macro")
        macro_f1 = f1_score(labels_all, predictions, average="macro")
        micro_f1 = f1_score(labels_all, predictions, average="micro")

        logger.info(
            f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
            f"Macro F1: {macro_f1:.3f}, Micro F1: {micro_f1:.3f}"
        )

        results = {
            "test/accuracy": accuracy,
            "test/precision": precision,
            "test/recall": recall,
            "test/macro_f1": macro_f1,
            "test/micro_f1": micro_f1,
        }

        return predictions, celltypes_labels, results

    def train(model: nn.Module, loader: DataLoader) -> None:
        """
        Train the model for one epoch.
        """
        model.train()
        total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
        total_error = 0.0
        log_interval = hyperparameter_defaults['log_interval']
        start_time = time.time()

        num_batches = len(loader)
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltypes"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=hyperparameter_defaults['amp']):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None,
                    MVC=hyperparameter_defaults['GEPC'],
                    ECS=hyperparameter_defaults['ecs_thres'] > 0,
                    CLS=config.CLS
                )

                masked_positions = input_values.eq(mask_value)  # the postions to predict

                loss = loss_mse = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                metrics_to_log = {"train/mse": loss_mse.item()}
                if explicit_zero_prob:
                    loss_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mlm_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_zero_log_prob
                    metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                if hyperparameter_defaults['GEPC']:
                    loss_gepc = criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                    loss = loss + loss_gepc
                    metrics_to_log.update({"train/mvc": loss_gepc.item()})
                if hyperparameter_defaults['GEPC'] and explicit_zero_prob:
                    loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_gepc_zero_log_prob
                    metrics_to_log.update(
                        {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                    )
                if hyperparameter_defaults['ecs_thres'] > 0:
                    loss_ecs = 10 * output_dict["loss_ecs"]
                    loss = loss + loss_ecs
                    metrics_to_log.update({"train/ecs": loss_ecs.item()})
                if config.CLS:
                    celltype_labels = torch.Tensor(celltype_labels).to(device)
                    loss_cls = criterion_loss(output_dict["cls_output"], celltype_labels)
                    loss = loss + loss_cls
                    metrics_to_log.update({"train/cls": loss_cls.item()})
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                loss = loss + hyperparameter_defaults['dab_weight'] * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                mre = masked_relative_error(
                    output_dict["mlm_output"], target_values, masked_positions
                )

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_gepc += loss_gepc.item() if hyperparameter_defaults['GEPC'] else 0.0
            total_error += mre.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                cur_gepc = total_gepc / log_interval if hyperparameter_defaults['GEPC'] else 0.0
                cur_error = total_error / log_interval
                # ppl = math.exp(cur_loss)
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                    + (f"gepc {cur_gepc:5.2f} |" if hyperparameter_defaults['GEPC'] else "")
                )
                total_loss = 0
                total_mse = 0
                total_gepc = 0
                total_error = 0
                start_time = time.time()

    # 模型创建
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        n_cls= num_types,
        vocab=vocab,
        dropout=hyperparameter_defaults['dropout'],
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=hyperparameter_defaults['GEPC'],
        do_dab=True,
        use_batch_labels=True,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=DSBN,
        n_input_bins=n_input_bins,
        ecs_threshold=hyperparameter_defaults['ecs_thres'],
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=hyperparameter_defaults['fast_transformer'],
        pre_norm=hyperparameter_defaults['pre_norm'],
    )
    model.to(device)
    criterion = masked_mse_loss
    criterion_loss = torch.nn.CrossEntropyLoss()
    criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyperparameter_defaults['lr'], eps=1e-4 if hyperparameter_defaults['amp'] else 1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=hyperparameter_defaults['schedule_ratio'])
    scaler = torch.cuda.amp.GradScaler(enabled=hyperparameter_defaults['amp'])
    best_val = 0
    best_model = None
    history_list =[]
    for epoch in range(1, hyperparameter_defaults['epochs'] + 1):
        epoch_start_time = time.time()
        train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=hyperparameter_defaults['batch_size'],
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
        )
        test_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=hyperparameter_defaults['batch_size'],
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False,
        )

        if hyperparameter_defaults['do_train']:
            train(
                model,
                loader=train_loader,
            )

        _,_, results = test(
            model,
            loader=test_loader
        )
        #         results = {
        #             "test/accuracy": accuracy,
        #             "test/precision": precision,
        #             "test/recall": recall,
        #             "test/macro_f1": macro_f1,
        #             "test/micro_f1": micro_f1,
        #         }
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        acc = results["test/accuracy"]
        f1 = results["test/macro_f1"]
        history_list.append(results)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/acc {acc:5.4f} |macro {f1:5.4f}"
        )
        logger.info("-" * 89)

        if acc > best_val:
            best_val = acc
            logger.info(f"Best model with score {best_val:5.4f}")
            torch.save(model, save_dir / f"model_best.pt")
        if epoch % 10 == 0:
            logger.info(f"Saving model to {save_dir}")
            torch.save(model, save_dir / f"model_e{epoch}.pt")
        scheduler.step()
        with open(save_dir / f"model_history.pkl",'wb')as file:
            pickle.dump(history_list, file)

for timepoint in batch_id_mapping.keys():
    print(timepoint)
    model_train_timepoint(timepoint,adata)
    print(timepoint+' end')
# model_train_timepoint('WT',adata)

