# 此部分用于实现无监督学习
import pickle
import os,re
import sys
from copy import copy,deepcopy

from loompy import loompy
from sklearn.metrics import f1_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import pandas as pd
import numpy as np
from geneformer import TranscriptomeTokenizer, DataCollatorForCellClassification
import time
from tqdm import trange
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA, FileTransferSpeed
import scanpy as sc
import loompy as lp
import datasets
import datetime
import random
import subprocess
import numpy as np
import pytz
import torch
from datasets import load_from_disk
from transformers import BertConfig, BertForMaskedLM, TrainingArguments, BertForSequenceClassification, get_scheduler, \
    Trainer
from geneformer import GeneformerPretrainer
from collections import Counter
from src.utils import collate_fn, compute_metrics

# 首先根据geneformer要求，将基因名称化为秩值编码
def unsupervised_train(model_path, dataset_path,tk_path, p_num_path, target_dir,config_dict):
    seed_num = config_dict['seed_num']
    random.seed(seed_num)
    seed_val = config_dict['seed_val']
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # set model parameters
    # model type
    model_type = config_dict['model_type']
    # max input size
    max_input_size = config_dict['max_input_size']
    # number of layers
    num_layers = config_dict['num_layers']
    # number of attention heads
    num_attn_heads = config_dict['num_attn_heads']
    # number of embedding dimensions
    num_embed_dim = config_dict['num_embed_dim']
    # intermediate size
    intermed_size = num_embed_dim * 2
    # activation function
    activ_fn = config_dict['activ_fn']
    # initializer range, layer norm, dropout
    initializer_range = config_dict['initializer_range']
    layer_norm_eps = config_dict['layer_norm_eps']
    attention_probs_dropout_prob = config_dict['attention_probs_dropout_prob']
    hidden_dropout_prob = config_dict['hidden_dropout_prob']
    # set training parameters
    # total number of examples in Genecorpus-30M after QC filtering:
    num_examples = config_dict['num_examples']
    # batch size for training and eval
    geneformer_batch_size = config_dict['batch_size']
    # max learning rate
    max_lr = config_dict['max_learning_rate']
    # learning schedule
    lr_schedule_fn = config_dict['lr_schedule_fn']
    # warmup steps
    warmup_steps = config_dict['warmup_steps']
    # number of epochs
    epochs = config_dict['epochs']
    # weight_decay
    weight_decay = config_dict['weight_decay']

    # output directories
    model_output_dir = target_dir
    training_output_dir = f'{target_dir}/train'
    logging_dir = f'{target_dir}/logs'
    # ensure not overwriting previously saved model
    model_output_file = os.path.join(model_output_dir, "pytorch_model.bin")
    if os.path.isfile(model_output_file) is True:
        raise Exception("Model already saved to this directory.")
    # make training and model output directories
    subprocess.call(f"mkdir {model_output_dir}", shell=True)
    subprocess.call(f"mkdir {training_output_dir}", shell=True)
    subprocess.call(f"mkdir {logging_dir}", shell=True)
    # 这里需要更改
    # load gene_ensembl_id:token dictionary (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/blob/main/token_dictionary.pkl)
    # with open("/home/wwd/codebox/Geneformer/geneformer/token_dictionary.pkl", "rb") as fp:
    #     token_dictionary = pickle.load(fp)
    with open(tk_path, "rb") as fp:
        token_dictionary = pickle.load(fp)
    # model configuration
    config = {
        "hidden_size": num_embed_dim,
        "num_hidden_layers": num_layers,
        "initializer_range": initializer_range,
        "layer_norm_eps": layer_norm_eps,
        "attention_probs_dropout_prob": attention_probs_dropout_prob,
        "hidden_dropout_prob": hidden_dropout_prob,
        "intermediate_size": intermed_size,
        "hidden_act": activ_fn,
        "max_position_embeddings": max_input_size,
        "model_type": model_type,
        "num_attention_heads": num_attn_heads * 2,
        "pad_token_id": token_dictionary.get("<pad>"),
        "vocab_size": len(token_dictionary),  # genes+2 for <mask> and <pad> tokens
    }
    # 设置基础的bert
    config = BertConfig(**config)
    model = BertForMaskedLM(config)
    # 使用预训练好的模型
    model = model.from_pretrained(model_path,
                                  config=config,
                                  ignore_mismatched_sizes=True)  # 如有结构变化
    model = model.train()

    # define the training arguments
    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "do_eval": False,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "per_device_train_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "save_strategy": "steps",
        "save_steps": np.floor(num_examples / geneformer_batch_size / 8),  # 8 saves per epoch
        "logging_steps": 1000,
        "output_dir": training_output_dir,
        "logging_dir": logging_dir,
    }
    training_args = TrainingArguments(**training_args)
    # 您将看到有关未使用某些预训练权重以及某些权重被随机初始化的警告。别担心，这是完全正常的！ BERT 模型的预训练头被丢弃，并替换为随机初始化的分类头。您将在序列分类任务中微调这个新模型头，将预训练模型的知识传递给它
    print("Starting training.")
    # 定义trainer,加入真实细胞数据训练，主要更改此文件
    print('begin')
    trainer = GeneformerPretrainer(
        model=model,
        args=training_args,
        # pretraining corpus (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/genecorpus_30M_2048.dataset)
        train_dataset=load_from_disk(dataset_path),
        # file of lengths of each example cell (e.g. https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/blob/main/genecorpus_30M_2048_lengths.pkl)
        example_lengths_file=p_num_path,
        token_dictionary=token_dictionary,
    )
    # train
    trainer.train()
    # save model
    trainer.save_model(model_output_dir)
    pass
def supervised_train(model, model_type, dataset_path, target_dir, num_epochs = 40, dataset_split = 0.8):
    if(model_type == "gene"):
        timepoints_dataset = load_from_disk(dataset_path)
        p_trainset = timepoints_dataset.select([i for i in range(0, round(len(timepoints_dataset) * dataset_split))])
        p_trainset = p_trainset.shuffle()
        p_evalset = timepoints_dataset.select(
            [i for i in range(round(len(timepoints_dataset) * dataset_split), len(timepoints_dataset))])
        os.makedirs(target_dir, exist_ok=True)
        geneformer_batch_size = 8
        training_args = {
            "learning_rate": 1e-5,
            "do_train": True,
            "do_eval": True,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_steps": round(len(p_trainset) / geneformer_batch_size / 10),
            "group_by_length": True,
            "length_column_name": "length",
            "disable_tqdm": False,
            "lr_scheduler_type": "linear",
            "warmup_steps": 500,
            "weight_decay": 0.001,
            "per_device_train_batch_size": geneformer_batch_size,
            "per_device_eval_batch_size": geneformer_batch_size,
            "num_train_epochs": num_epochs,
            "load_best_model_at_end": True,
            "output_dir": target_dir,
        }
        training_args_init = TrainingArguments(**training_args)
        # create the trainer
        trainer = Trainer(
            model=model,
            args=training_args_init,
            data_collator=DataCollatorForCellClassification(),
            train_dataset=p_trainset,
            eval_dataset=p_evalset,
            compute_metrics=compute_metrics
        )
        # train the cell type classifier
        trainer.train()
        trainer.save_model()
        pass
    elif(model_type == "spatial"):
        cell_dataset = datasets.Dataset.load_from_disk(dataset_path)
        cell_dataset = cell_dataset.shuffle(123)
        # 数据集划分 为防止数据泄露，先进行划分
        p_trainset = cell_dataset.select([i for i in range(0, round(len(cell_dataset) * dataset_split))])
        p_evalset = cell_dataset.select([i for i in range(round(len(cell_dataset) * dataset_split), round(len(cell_dataset)))])
        train_loader = DataLoader(p_trainset, batch_size=8, collate_fn=collate_fn, shuffle=True)
        test_loader = DataLoader(p_evalset, batch_size=16, collate_fn=collate_fn, shuffle=True)
        os.makedirs(target_dir, exist_ok=True)
        # 定义优化器和损失函数
        optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-5)  # weight_decay参数控制L2正则化
        # 设置学习率调度器
        num_training_steps = num_epochs * len(train_loader)
        criterion = torch.nn.CrossEntropyLoss()
        # 学习率调度器
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps
        )
        # 设定 early stopping 参数
        accuracy_list = []
        f1_list = []
        test_accuracys = []
        test_accuracy = 0
        best_accuracy = 0
        for epoch in range(num_epochs):
            # 训练模型
            model.train()
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0
            progress_bar = tqdm(train_loader)
            correct = 0
            total = 0
            all_labels = []
            all_predictions = []
            for batch in progress_bar:
                input_ids, labels, spatial, lengths = [item.to('cuda') for item in batch]
                total += labels.size(0)
                # 前向传播
                input_ids = input_ids.long()
                # 注意力掩码，确保模型忽略填充的部分
                attention_mask = (input_ids != 0).long()
                spatial = spatial.float()

                # 前向传播，加入空间信息
                outputs, _, _, l2_loss = model(epoch, input_ids=input_ids, spatial_info=spatial,
                                                       attention_mask=attention_mask, gene_length=lengths)
                # outputs = spatial_model(input_ids=input_ids, spatial_info=spatial)
                loss = criterion(outputs, labels)
                loss = loss + l2_loss * 0.01
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_value_(spatial_model.parameters(), clip_value=0.1)
                optimizer.step()
                lr_scheduler.step()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                # 更新进度条和损失
                epoch_loss += loss.item()
                progress_bar.set_description(f"Loss: {loss.item():.4f}")
                # 收集所有标签和预测
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

            accuracy = correct / total
            # 计算 F1 分数
            f1 = f1_score(all_labels, all_predictions, average='macro')
            print(
                f"Train Epoch Loss: {epoch_loss / len(train_loader):.4f}, Train Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
            # 测试模型
            model.eval()
            correct = 0
            total = 0
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids, labels, spatial, lengths = [item.to('cuda') for item in batch]
                    # 注意力掩码，确保模型忽略填充的部分
                    attention_mask = (input_ids != 0).long()
                    input_ids = input_ids.long()
                    spatial = spatial.float()
                    # 前向传播
                    outputs, _, _, _ = model(epoch, input_ids=input_ids, spatial_info=spatial,
                                                     attention_mask=attention_mask, gene_length=lengths)
                    _, predicted = torch.max(outputs, 1)

                    # 统计正确的预测
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # 收集所有标签和预测
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
            accuracy = correct / total
            if (test_accuracy < accuracy):
                test_accuracy = accuracy
            # 计算 F1 分数
            f1 = f1_score(all_labels, all_predictions, average='macro')
            print(f"Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
            accuracy_list.append(accuracy)
            f1_list.append(f1)
            # 检查是否是当前最佳模型
            if accuracy > best_accuracy:
                print(
                    f"New best model found! Accuracy improved from {best_accuracy:.4f} to {accuracy:.4f}. Saving model...")
                best_accuracy = accuracy
                no_improvement_epochs = 0  # 重置无提升计数
                # 保存当前最佳模型
                torch.save(model,
                           target_dir + "/bert_with_spatial_model_best.pth")
            # 每10个一保存
            if ((epoch + 1) % 10 == 0):
                # 每十个epoch保存模型
                torch.save(model,
                           target_dir + "/bert_with_spatial_model_" + str(
                               epoch + 1) + ".pth")
                print(target_dir + "/bert_with_spatial_model_" + str(
                    epoch + 1) + ".pth saved")
            test_accuracys.append(test_accuracy)
        with open(target_dir + "/history.txt", 'w') as file:
            for i in range(len(accuracy_list)):
                file.write(str(accuracy_list[i]))
                file.write(' ')
                file.write(str(f1_list[i]))
                file.write('\n')
        print(test_accuracys)
        pass
    pass

