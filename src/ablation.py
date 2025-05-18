# 此文件用于模型的比较
import json
import os, re
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
import pickle
from collections import Counter
import seaborn as sns
sns.set()
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk, concatenate_datasets
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification, get_scheduler, DebertaForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments
from geneformer import DataCollatorForCellClassification
from tqdm import tqdm
import torch
import datasets
import random
import subprocess
import numpy as np
from src.models import BertWithSpatialInfo_spatial3
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    # 评价
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
        'accuracy': acc,
        'macro_f1': macro_f1
    }
import datasets
def geneModelRawCelltype(dataset_path='./dataSets/all_cell.datasets' ,pretrained_path="./models/pretrained_model",output_path='./models/gene_raw2/'):
    # 基础基因模型 预测细胞种类
    timepoints = ['0h','12h','1.5d','3d','5d','10d','WT']
    celltype_label = {'gut': 0, 'Nb2': 1, 'cathepsin_cells': 2, 'epidermal': 3, 'muscle': 4, 'parenchymal': 5, 'protonephridia': 6, 'pharynx': 7, 'neural': 8}
    # with open('./dataSets/gene_tk.pkl', 'rb') as f:
    #     gene2id = pickle.load(f)
    # 数据读取
    dataSet_all = datasets.Dataset.load_from_disk(dataset_path)
    dataSet_all = dataSet_all.rename_column('cell_genes', 'input_ids')
    dataSet_all = dataSet_all.rename_column('cell_type', 'label')
    def label_map(example):
        example['label'] = celltype_label[example['label']]
        example['input_ids'] = example['input_ids'][:2048] # 超过上界的直接去掉
        random.shuffle(example['input_ids'])
        example['length'] = len(example['input_ids'])
        return example
    # 数据分割与处理
    for timepoint in timepoints:
        timepoints_dataset = dataSet_all.filter(lambda x: x['timepoint'] == timepoint)
        print(timepoints_dataset)
        timepoints_dataset = timepoints_dataset.map(label_map,num_proc=16)
        timepoints_dataset = timepoints_dataset.shuffle()
        # 加载预训练好的模型，为了与空间模型相比较，选择经过无监督训练的模型
        gene_model = BertForSequenceClassification.from_pretrained(pretrained_path,
                                                              num_labels=len(celltype_label.keys()),
                                                              output_attentions=False,
                                                              output_hidden_states=False,ignore_mismatched_sizes=True).to('cuda')

        # 重新初始化模型参数
        def initialize_weights(model):
            for name, param in model.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)  # Bias 初始化为0
                elif 'weight' in name:
                    if 'LayerNorm' in name:
                        nn.init.normal_(param, mean=1.0, std=0.02)  # LayerNorm 的权重使用正态分布初始化
                    else:
                        nn.init.xavier_normal_(param)  # 其他权重使用 Xavier 正态初始化

        # 重新初始化模型参数
        #initialize_weights(gene_model)
        p_trainset = timepoints_dataset.select([i for i in range(0, round(len(timepoints_dataset) * 0.8))])
        p_evalset = timepoints_dataset.select(
            [i for i in range(round(len(timepoints_dataset) * 0.8), len(timepoints_dataset))])
        output_dir = output_path +timepoint+"/"
        os.makedirs(output_dir, exist_ok=True)
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
            "num_train_epochs": 40,
            "load_best_model_at_end": True,
            "output_dir": output_dir,

        }
        training_args_init = TrainingArguments(**training_args)
        # create the trainer
        trainer = Trainer(
            model=gene_model,
            args=training_args_init,
            data_collator=DataCollatorForCellClassification(),
            train_dataset=p_trainset,
            eval_dataset=p_evalset,
            compute_metrics=compute_metrics
        )
        # train the cell type classifier
        trainer.train()
        trainer.save_model()
def geneModelAugCelltype(dataset_path='./models/all_cell.datasets',pretrained_path="./models/pretrained_model",output_path='./models/gene_aug/'):
    timepoints = ['0h', '12h', '1.5d', '3d', '5d', '10d', 'WT']
    celltype_label = {'gut': 0, 'Nb2': 1, 'cathepsin_cells': 2, 'epidermal': 3, 'muscle': 4, 'parenchymal': 5,
                      'protonephridia': 6, 'pharynx': 7, 'neural': 8}
    with open('./tk_dict/gene_tk.pkl', 'rb') as f:
        gene2id = pickle.load(f)
    # 数据读取
    dataSet_all = datasets.Dataset.load_from_disk(dataset_path)
    dataSet_all = dataSet_all.rename_column('cell_genes', 'input_ids')
    dataSet_all = dataSet_all.rename_column('cell_type', 'label')

    def label_map(example):
        example['label'] = celltype_label[example['label']]
        random.shuffle(example['input_ids'])
        example['input_ids'] = example['input_ids'][:2048]
        example['length'] = len(example['input_ids'])
        return example
    # 数据分割与处理
    for timepoint in timepoints:
        timepoints_dataset = dataSet_all.filter(lambda x: x['timepoint'] == timepoint)
        print(timepoints_dataset)
        timepoints_dataset = timepoints_dataset.shuffle()
        # 加载预训练好的模型，为了与空间模型相比较，选择经过无监督训练的模型
        gene_model = BertForSequenceClassification.from_pretrained(pretrained_path,
                                                                   num_labels=len(celltype_label.keys()),
                                                                   output_attentions=False,
                                                                   output_hidden_states=False,
                                                                   ignore_mismatched_sizes=True).to('cuda')

        # 重新初始化模型参数
        def initialize_weights(model):
            for name, param in model.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)  # Bias 初始化为0
                elif 'weight' in name:
                    if 'LayerNorm' in name:
                        nn.init.normal_(param, mean=1.0, std=0.02)  # LayerNorm 的权重使用正态分布初始化
                    else:
                        nn.init.xavier_normal_(param)  # 其他权重使用 Xavier 正态初始化

        # 重新初始化模型参数
        #initialize_weights(gene_model)
        p_trainset = timepoints_dataset.select([i for i in range(0, round(len(timepoints_dataset) * 0.8))])
        p_cell_aug = []
        for cell in p_trainset:
            aug_number = max(1,int(len(cell['input_ids']) / 200))
            for _ in range(aug_number):
                num_mask = random.randint(0, int(len(cell['input_ids']) * 0.35))
                num_genes = min(2048, max(200, len(cell['input_ids']) - num_mask))
                random.shuffle(cell['input_ids'])
                p_cell_aug.append({'label':cell['label'], 'input_ids':cell['input_ids'][:num_genes]})
            p_cell_aug.append(cell)
        p_trainset = datasets.Dataset.from_list(p_cell_aug)
        p_trainset = p_trainset.map(label_map,num_proc=16)
        p_trainset = p_trainset.shuffle()
        p_evalset = timepoints_dataset.select(
            [i for i in range(round(len(timepoints_dataset) * 0.8), len(timepoints_dataset))])
        p_evalset = p_evalset.map(label_map,num_proc=16)
        output_dir = output_path + timepoint + "/"
        os.makedirs(output_dir, exist_ok=True)
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
            "num_train_epochs": 40,
            "load_best_model_at_end": True,
            "output_dir": output_dir,
        }
        training_args_init = TrainingArguments(**training_args)
        # create the trainer
        trainer = Trainer(
            model=gene_model,
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
def SpatialModelRaw(pretrained_path="./models/pretrained_model",output_path='./models/spatial_raw/'):
    # 基础基因模型 预测细胞种类
    cell_Dataset = datasets.Dataset.load_from_disk('./dataSets/all_cell.datasets')
    cell_Dataset = cell_Dataset.rename_column('cell_genes', 'input_ids')
    timepoints = list(dict(Counter(cell_Dataset['timepoint'])).keys())
    celltype_label  = {'gut': 0, 'Nb2': 1, 'cathepsin_cells': 2, 'epidermal': 3, 'muscle': 4, 'parenchymal': 5, 'protonephridia': 6, 'pharynx': 7, 'neural': 8}
    timepoints_datasets = {}
    def label_map(example):
        example['label'] = celltype_label[example['label']]
        example['input_ids'] = example['input_ids'][:2048]
        random.shuffle(example['input_ids'])
        example['length'] = len(example['input_ids'])
        return example
    for timepoint in timepoints[:]:
        timepoints_datasets[timepoint] = cell_Dataset.filter(lambda x: x['timepoint'] == timepoint)
        os.makedirs(output_path + timepoint + '/', exist_ok=True)
        cell_dataset = timepoints_datasets[timepoint]
        cell_dataset = cell_dataset.rename_column('cell_type', 'label')
        cell_dataset = cell_dataset.map(label_map, num_proc=16)
        cell_dataset = cell_dataset.filter(lambda x: x['length'] >= 50)
        # 计算X,Y 的平均值，方差
        Xs = []
        Ys = []
        for cell in cell_dataset:
            Xs.append(cell['spatial'][0])
            Ys.append(cell['spatial'][1])
        X_mean = np.mean(Xs)  # 计算平均值
        Y_mean = np.mean(Ys)  # 计算平均值
        Y_std = np.std(Ys)  # 计算标准差
        # 找到左右两只涡虫的分割点
        # 先聚类
        # 此数据集要分成两个类
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit([[item, 1] for item in Xs])
        # 获取聚类标签和中心
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        # 根据中心的 x 坐标确定左右类
        if centers[0][0] < centers[1][0]:
            left_class_label = 0
            right_class_label = 1
        else:
            left_class_label = 1
            right_class_label = 0

        # 创建新的 labels 列
        new_labels = []
        for label in labels:
            if label == left_class_label:
                new_labels.append(0)  # 左类
            else:
                new_labels.append(1)  # 右类
        cell_dataset = cell_dataset.add_column('space', new_labels)
        new_Data_left = []
        new_Data_right = []
        Xs_left = []
        Xs_right = []
        for cell in cell_dataset:
            if (cell['space'] == 1):
                # 右部的涡虫
                new_Data_right.append({'input_ids': cell['input_ids'], 'label': cell['label'], 'length': cell['length'],
                                       'spatial': [cell['spatial'][0], cell['spatial'][1]],
                                       'spatial_raw': [cell['spatial'][0], cell['spatial'][1]], 'space': 1})
                Xs_right.append(cell['spatial'][0])
            else:
                # 左部的涡虫
                new_Data_left.append({'input_ids': cell['input_ids'], 'label': cell['label'], 'length': cell['length'],
                                      'spatial': [cell['spatial'][0], cell['spatial'][1]],
                                      'spatial_raw': [cell['spatial'][0], cell['spatial'][1]], 'space': 0})
                Xs_left.append(cell['spatial'][0])
        X_mean_left = np.mean(Xs_left)  # 计算左
        X_std_left = np.std(Xs_left)
        X_mean_right = np.mean(Xs_right)  # 计算右
        X_std_right = np.std(Xs_right)
        epsilon = 1e-8
        for cell in new_Data_left:
            cell['spatial'][0] = (cell['spatial'][0] - X_mean_left) / (X_std_left + epsilon)
        for cell in new_Data_right:
            cell['spatial'][0] = (cell['spatial'][0] - X_mean_right) / (X_std_right + epsilon)
        new_Data = new_Data_left + new_Data_right
        cell_dataset = datasets.Dataset.from_list(new_Data)
        # 计算新的数据集
        Xs = []
        for cell in cell_dataset:
            Xs.append(cell['spatial'][0])
            Ys.append(cell['spatial'][1])
        X_mean = np.mean(Xs)  # 计算平均值
        Y_mean = np.mean(Ys)  # 计算平均值
        Y_std = np.std(Ys)  # 计算标准差
        X_std = np.std(Xs)
        # 避免除以零
        epsilon = 1e-8

        def normalized_spatial(example):
            x = example['spatial'][0]
            y = example['spatial'][1]
            normalized_x = (x - X_mean) / (X_std + epsilon)
            normalized_y = (y - Y_mean) / (Y_std + epsilon)
            example['spatial'] = [normalized_x, normalized_y]
            return example
            pass

        cell_dataset = cell_dataset.map(normalized_spatial, num_proc=16)
        cell_dataset = cell_dataset.shuffle(123)
        # 数据集划分 为防止数据泄露，先进行划分
        p_trainset = cell_dataset.select([i for i in range(0, round(len(cell_dataset) * 0.8))])
        p_evalset = cell_dataset.select([i for i in range(round(len(cell_dataset) * 0.8), round(len(cell_dataset)))])
        # 数据准备
        p_trainset = p_trainset.shuffle()
        p_evalset = p_evalset.shuffle()
        os.makedirs(output_path+timepoint+'/',exist_ok=True)
        # 模型加载
        #gene_model_path = "/home/wwd/codebox/Geneformer/new/planaria2/results/model_new/250314_geneformer_CellClassifier_L2048_B12_LR1e-05_LSlinear_WU500_E120_Oadamw_F4_new_func/timepoint/best_model.pth"
        spatial_model =  BertWithSpatialInfo_spatial3.from_pretrained(pretrained_path,
                                                                   num_labels=len(celltype_label.keys()),
                                                                   output_attentions=False,
                                                                   output_hidden_states=False,
                                                                   ignore_mismatched_sizes=True).to('cuda')

        # 重新初始化模型参数
        def initialize_weights(model):
            for name, param in model.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)  # Bias 初始化为0
                elif 'weight' in name:
                    if 'LayerNorm' in name:
                        nn.init.normal_(param, mean=1.0, std=0.02)  # LayerNorm 的权重使用正态分布初始化
                    nn.init.normal_(param, mean=1.0, std=0.02)
        # 重新初始化模型参数
        #initialize_weights(spatial_model)
        # 定义 collate_fn 函数
        def collate_fn(batch):
            input_ids = []
            lengths = []
            spatial = []
            for item in batch:
                random.shuffle(item['input_ids'])
                input_ids.append(torch.tensor(item['input_ids']))
                lengths.append([item['length']])
                spatial.append(torch.tensor(item['spatial']))
            lengths = torch.tensor(lengths)
            labels = torch.tensor([item['label'] for item in batch])
            spatial = torch.stack(spatial)  # 转换为tensor
            # 对 input_ids 进行填充，使长度一致
            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
            return input_ids_padded, labels, spatial, lengths
        train_loader = DataLoader(p_trainset, batch_size= 8, collate_fn=collate_fn, shuffle=True)
        test_loader = DataLoader(p_evalset, batch_size= 16, collate_fn=collate_fn, shuffle=True)
        # 定义优化器和损失函数
        optimizer = optim.AdamW(spatial_model.parameters(), lr=3e-5, weight_decay=1e-5)  # weight_decay参数控制L2正则化
        # 设置学习率调度器
        num_epochs = 40
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
            spatial_model.train()
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
                outputs, _, _, l2_loss = spatial_model(epoch, input_ids=input_ids, spatial_info=spatial,
                                                 attention_mask=attention_mask, gene_length=lengths)
                # outputs = spatial_model(input_ids=input_ids, spatial_info=spatial)
                loss = criterion(outputs, labels)
                loss = loss + l2_loss*0.01
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
            spatial_model.eval()
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
                    outputs, _, _, _ = spatial_model(epoch, input_ids=input_ids, spatial_info=spatial,
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
                torch.save(spatial_model,
                           output_path + timepoint + "/bert_with_spatial_model_best.pth")
            # 每10个一保存
            if ((epoch + 1) % 10 == 0):
                # 每十个epoch保存模型
                torch.save(spatial_model,
                           output_path + timepoint + "/bert_with_spatial_model_" + str(
                               epoch + 1) + ".pth")
                print(output_path + timepoint + "/bert_with_spatial_model_" + str(
                    epoch + 1) + ".pth saved")
            test_accuracys.append(test_accuracy)
        with open(output_path + timepoint + "/history.txt", 'w') as file:
            for i in range(len(accuracy_list)):
                file.write(str(accuracy_list[i]))
                file.write(' ')
                file.write(str(f1_list[i]))
                file.write('\n')
        print(test_accuracys)
    pass
def SpatialModelAug(pretrained_path="./models/pretrained_model",output_path='./models/spatial_aug/'):
    cell_Dataset = datasets.Dataset.load_from_disk('./dataSets/all_cell.datasets')
    cell_Dataset = cell_Dataset.rename_column('cell_genes', 'input_ids')
    timepoints = list(dict(Counter(cell_Dataset['timepoint'])).keys())
    celltype_label  = {'gut': 0, 'Nb2': 1, 'cathepsin_cells': 2, 'epidermal': 3, 'muscle': 4, 'parenchymal': 5, 'protonephridia': 6, 'pharynx': 7, 'neural': 8}
    timepoints_datasets = {}
    def label_map(example):
        example['label'] = celltype_label[example['label']]
        example['input_ids'] = example['input_ids']
        random.shuffle(example['input_ids'])
        example['length'] = len(example['input_ids'])
        return example
    for timepoint in timepoints:
        timepoints_datasets[timepoint] = cell_Dataset.filter(lambda x: x['timepoint'] == timepoint)
        os.makedirs(output_path + timepoint + '/', exist_ok=True)
        cell_dataset = timepoints_datasets[timepoint]
        cell_dataset = cell_dataset.rename_column('cell_type','label')
        cell_dataset = cell_dataset.map(label_map,num_proc=16)
        cell_dataset = cell_dataset.filter(lambda x: x['length'] >= 50)
        # 计算X,Y 的平均值，方差
        Xs = []
        Ys = []
        for cell in cell_dataset:
            Xs.append(cell['spatial'][0])
            Ys.append(cell['spatial'][1])
        X_mean = np.mean(Xs) # 计算平均值
        Y_mean = np.mean(Ys)  # 计算平均值
        Y_std = np.std(Ys)    # 计算标准差
        # 找到左右两只涡虫的分割点
        # 先聚类
        # 此数据集要分成两个类
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit([[item,1] for item in Xs])
        # 获取聚类标签和中心
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        # 根据中心的 x 坐标确定左右类
        if centers[0][0] < centers[1][0]:
            left_class_label = 0
            right_class_label = 1
        else:
            left_class_label = 1
            right_class_label = 0

        # 创建新的 labels 列
        new_labels = []
        for label in labels:
            if label == left_class_label:
                new_labels.append(0)  # 左类
            else:
                new_labels.append(1)  # 右类
        cell_dataset = cell_dataset.add_column('space',new_labels)
        new_Data_left =[]
        new_Data_right = []
        Xs_left =[]
        Xs_right =[]
        for cell in cell_dataset:
            if(cell['space']==1 ):
                # 右部的涡虫
                new_Data_right.append({'input_ids':cell['input_ids'],'label':cell['label'],'length':cell['length'],
                                 'spatial':[cell['spatial'][0], cell['spatial'][1]],'spatial_raw':[cell['spatial'][0], cell['spatial'][1]],'space':1})
                Xs_right.append(cell['spatial'][0])
            else:
                # 左部的涡虫
                new_Data_left.append({'input_ids': cell['input_ids'], 'label': cell['label'], 'length': cell['length'],
                                 'spatial': [cell['spatial'][0], cell['spatial'][1]],'spatial_raw':[cell['spatial'][0], cell['spatial'][1]],'space':0})
                Xs_left.append(cell['spatial'][0])
        X_mean_left = np.mean(Xs_left) # 计算左
        X_std_left = np.std(Xs_left)
        X_mean_right = np.mean(Xs_right) # 计算右
        X_std_right = np.std(Xs_right)
        epsilon = 1e-8
        for cell in new_Data_left:
            cell['spatial'][0] = (cell['spatial'][0] - X_mean_left) / (X_std_left  + epsilon)
        for cell in new_Data_right:
            cell['spatial'][0] = (cell['spatial'][0] - X_mean_right) / (X_std_right  + epsilon)
        new_Data = new_Data_left + new_Data_right
        cell_dataset = datasets.Dataset.from_list(new_Data)
        # 计算新的数据集
        Xs = []
        for cell in cell_dataset:
            Xs.append(cell['spatial'][0])
            Ys.append(cell['spatial'][1])
        X_mean = np.mean(Xs) # 计算平均值
        Y_mean = np.mean(Ys)  # 计算平均值
        Y_std = np.std(Ys)    # 计算标准差
        X_std = np.std(Xs)
        # 避免除以零
        epsilon = 1e-8

        def spatial_aug_fc(spatial_info, noise_scale=0.01):
            noise = np.random.uniform(low=-noise_scale, high=noise_scale, size=spatial_info.shape)
            augmented_spatial_info = spatial_info + noise
            return augmented_spatial_info
        def normalized_spatial(example):
            x = example['spatial'][0]
            y = example['spatial'][1]
            normalized_x = (x - X_mean) / (X_std + epsilon)
            normalized_y = (y - Y_mean) / (Y_std + epsilon)
            example['spatial'] = [normalized_x ,normalized_y]
            return example
            pass
        cell_dataset = cell_dataset.map(normalized_spatial,num_proc=16)
        cell_dataset = cell_dataset.shuffle(123)
        # 数据集划分 为防止数据泄露，先进行划分
        train_dataset = cell_dataset.select([i for i in range(0, round(len(cell_dataset) * 0.8))])
        test_dataset = cell_dataset.select([i for i in range(round(len(cell_dataset) * 0.8),round(len(cell_dataset)))])
        p_cell_aug = []
        for cell in train_dataset:
            aug_number = max(1, int(len(cell['input_ids']) / 200))
            for _ in range(aug_number):
                num_mask = random.randint(0, int(len(cell['input_ids']) * 0.35))
                num_genes = min(2048, max(200, len(cell['input_ids']) - num_mask)) # 最多2048，最少200个，如果不足，就不删减
                random.shuffle(cell['input_ids'])
                spatial_aug = spatial_aug_fc(np.array(cell['spatial']))
                p_cell_aug.append({'label':cell['label'], 'input_ids':cell['input_ids'][:num_genes],'spatial':np.array(spatial_aug),'length':len(cell['input_ids'][:num_genes])})
            p_cell_aug.append(cell)
        train_dataset = datasets.Dataset.from_list(p_cell_aug)
        train_dataset = train_dataset.shuffle()
        test_dataset = test_dataset.shuffle()
        spatial_model =  BertWithSpatialInfo_spatial3.from_pretrained(pretrained_path,
                                                                   num_labels=len(celltype_label),
                                                                   output_attentions=False,
                                                                   output_hidden_states=False,
                                                                   ignore_mismatched_sizes=True).to('cuda')
        def initialize_weights(model):
            for name, param in model.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)  # Bias 初始化为0
                elif 'weight' in name:
                    if 'LayerNorm' in name:
                        nn.init.normal_(param, mean=1.0, std=0.02)  # LayerNorm 的权重使用正态分布初始化
                    else:
                        nn.init.xavier_normal_(param)  # 其他权重使用 Xavier 正态初始化

        # 重新初始化模型参数
        #initialize_weights(spatial_model)
        def collate_fn(batch):
            input_ids = []
            lengths = []
            spatial = []
            for item in batch:
                random.shuffle(item['input_ids'])
                input_ids.append(torch.tensor(item['input_ids'][:2048]))
                lengths.append([item['length']])
                spatial.append(torch.tensor(item['spatial']))
            lengths = torch.tensor(lengths)
            labels = torch.tensor([item['label'] for item in batch])
            spatial = torch.stack(spatial)  # 转换为tensor
            # 对 input_ids 进行填充，使长度一致
            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
            return input_ids_padded, labels, spatial, lengths
        train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)
        # 定义优化器和损失函数
        optimizer = optim.AdamW(spatial_model.parameters(), lr=3e-5, weight_decay=1e-3)  # weight_decay参数控制L2正则化
        criterion = torch.nn.CrossEntropyLoss()
        # 设置学习率调度器
        num_epochs = 40
        num_training_steps = num_epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps
        )
        best_accuracy = 0.0  # 最佳准确率
        accuracy_list=[]
        f1_list =[]
        for epoch in range(num_epochs):
            # 训练模型
            spatial_model.train()
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0
            progress_bar = tqdm(train_loader)
            correct = 0
            total = 0
            all_labels = []
            all_predictions = []
            for batch in progress_bar:
                input_ids, labels, spatial,lengths = [item.to('cuda') for item in batch]
                total += labels.size(0)
                # 前向传播
                input_ids = input_ids.long()
                # 注意力掩码，确保模型忽略填充的部分
                attention_mask = (input_ids != 0).long()
                spatial = spatial.float()
                # 前向传播，加入空间信息
                outputs,_,_,l2_loss = spatial_model(epoch,input_ids=input_ids, spatial_info=spatial, attention_mask=attention_mask,gene_length = lengths)

                loss = criterion(outputs, labels)
                loss = loss +l2_loss*0.03
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
            print(f"Train Epoch Loss: {epoch_loss / len(train_loader):.4f}, Train Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
            # 测试模型
            spatial_model.eval()
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
                    outputs,_,_,_ = spatial_model(epoch,input_ids=input_ids, spatial_info=spatial, attention_mask=attention_mask,gene_length = lengths)
                    # outputs = spatial_model(input_ids=input_ids, spatial_info=spatial)
                    # print(outputs)
                    _, predicted = torch.max(outputs, 1)

                    # 统计正确的预测
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # 收集所有标签和预测
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
            accuracy = correct / total

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
                torch.save(spatial_model,output_path+timepoint+ "/bert_with_spatial_model_best.pth")

            if((epoch+1)%10 == 0):
                # 每十个epoch保存模型
                torch.save(spatial_model,output_path+timepoint+ "/bert_with_spatial_model_"+str(epoch+1)+".pth")
                print(output_path+timepoint+ "/bert_with_spatial_model_"+str(epoch+1)+".pth saved")

        with open(output_path+timepoint+ "/history.txt",'w')as file:
            for i in range(len(accuracy_list)):
                file.write(str(accuracy_list[i]))
                file.write(' ')
                file.write(str(f1_list[i]))
                file.write('\n')

    pass
def getfolder(target_path):
    # 构建目标路径
    file_path = target_path

    # 获取该路径下的所有文件夹
    folders = [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]

    # 筛选出以 'checkpoints-' 开头的文件夹，并提取其中的数字
    checkpoint_folders = []
    for folder in folders:
        if folder.startswith('checkpoint-'):
            # 提取数字部分并转换为整数
            try:
                num = int(folder.split('-')[1])
                checkpoint_folders.append((num, folder))
            except ValueError:
                continue  # 如果文件夹名后面的数字不合法，则跳过

    # 找到数字最大的一项
    if checkpoint_folders:
        max_num, max_folder = max(checkpoint_folders, key=lambda x: x[0])
        return os.path.join(target_path, max_folder)
    else:
        print("No checkpoints found.")
def getValues(target_path,type = 'spatial'):
    # 获取历史accuracy和f1分数
    timepoints = ['0h','12h','1.5d','3d','5d','10d','WT']
    history_best = {}
    if type == 'spatial':
        for timepoint in timepoints:
            history_best[timepoint] = []
            history_item = []
            file_path = target_path+timepoint+'/history.txt'
            with open(file_path,'r') as file:
                lines = file.readlines()
            for line in lines:
                line = line.strip('\n')
                line = line.split(' ')
                history_item.append([float(line[0]),float(line[1])])
            sorted_history_item = sorted(history_item, key=lambda x: x[0],reverse=True)
            history_best[timepoint] = [sorted_history_item[0]]
        print(history_best)
    if type == 'gene':
        for timepoint in timepoints:
            history_best[timepoint] = []
            history_item = []
            file_path = getfolder(target_path,timepoint) +'/trainer_state.json'
            # 读取并解析 JSON 文件
            with open(file_path, 'r') as file:
                data = json.load(file)
            history_item = []
            for item in data['log_history']:
                if 'eval_accuracy' in item and 'eval_macro_f1' in item:
                    history_item.append([
                      item['eval_accuracy'],
                         item['eval_macro_f1']
                    ])
            sorted_history_item = sorted(history_item, key=lambda x: x[0], reverse=True)
            history_best[timepoint] = [sorted_history_item[0]]
        print(history_best)

def getfolder2(target_path, timepoint):
    # 构建目标路径
    file_path = os.path.join(target_path, timepoint)

    # 获取该路径下的所有文件夹
    folders = [f for f in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, f))]

    # 筛选出以 'checkpoints-' 开头的文件夹，并提取其中的数字
    checkpoint_folders = []
    for folder in folders:
        if folder.startswith('checkpoint-'):
            # 提取数字部分并转换为整数
            try:
                num = int(folder.split('-')[1])
                checkpoint_folders.append((num, folder))
            except ValueError:
                continue  # 如果文件夹名后面的数字不合法，则跳过

    # 找到数字最大的一项
    if checkpoint_folders:
        max_num, max_folder = max(checkpoint_folders, key=lambda x: x[0])
        return os.path.join(target_path, timepoint,max_folder)
    else:
        print("No checkpoints found.")

def ModelIntegration(model_type = 'Bert',pretrained_path= "./models/pretrained_model",output_path='./models/time_spatial/',aug=True):
    # 完整的训练过程，先用时期标签训练，在预测标签分类
    # 先进行数据集划分防止数据泄露
    # 为保证数据分布均衡，分时期划分0.8为训练集，0.2为测试集
    cell_Dataset = datasets.Dataset.load_from_disk('./dataSets/all_cell.datasets')
    cell_Dataset = cell_Dataset.rename_column('cell_genes', 'input_ids')
    timepoints = list(dict(Counter(cell_Dataset['timepoint'])).keys())
    celltype_label = {'gut': 0, 'Nb2': 1, 'cathepsin_cells': 2, 'epidermal': 3, 'muscle': 4, 'parenchymal': 5,
                      'protonephridia': 6, 'pharynx': 7, 'neural': 8}
    timepoint_label = {'0h' : 0, '12h' : 1, '1.5d' : 2, '3d' : 3, '5d' : 4, '10d' : 5,'WT' : 6}
    timepoints_datasets = {}
    train_dataset_all = []
    test_dataset_all = []
    print('begin')
    def label_map(example):
        example['type_label'] = celltype_label[example['type_label']]
        example['label'] = timepoint_label[example['label']]
        example['input_ids'] = example['input_ids']
        random.shuffle(example['input_ids'])
        example['length'] = len(example['input_ids'])
        return example

    for timepoint in timepoints:
        timepoints_datasets[timepoint] = cell_Dataset.filter(lambda x: x['timepoint'] == timepoint)
        os.makedirs(output_path + timepoint + '/', exist_ok=True)
        cell_dataset = timepoints_datasets[timepoint]
        cell_dataset = cell_dataset.rename_column('cell_type', 'type_label')
        cell_dataset = cell_dataset.rename_column('timepoint', 'label')
        cell_dataset = cell_dataset.map(label_map, num_proc=16)
        cell_dataset = cell_dataset.filter(lambda x: x['length'] >= 50)
        # 计算X,Y 的平均值，方差
        Xs = []
        Ys = []
        for cell in cell_dataset:
            Xs.append(cell['spatial'][0])
            Ys.append(cell['spatial'][1])
        X_mean = np.mean(Xs)  # 计算平均值
        Y_mean = np.mean(Ys)  # 计算平均值
        Y_std = np.std(Ys)  # 计算标准差
        # 找到左右两只涡虫的分割点
        # 先聚类
        # 此数据集要分成两个类
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit([[item, 1] for item in Xs])
        # 获取聚类标签和中心
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        # 根据中心的 x 坐标确定左右类
        if centers[0][0] < centers[1][0]:
            left_class_label = 0
            right_class_label = 1
        else:
            left_class_label = 1
            right_class_label = 0

        # 创建新的 labels 列
        new_labels = []
        for label in labels:
            if label == left_class_label:
                new_labels.append(0)  # 左类
            else:
                new_labels.append(1)  # 右类
        cell_dataset = cell_dataset.add_column('space', new_labels)
        new_Data_left = []
        new_Data_right = []
        Xs_left = []
        Xs_right = []
        for cell in cell_dataset:
            if (cell['space'] == 1):
                # 右部的涡虫
                new_Data_right.append({'input_ids': cell['input_ids'], 'label': cell['label'], 'type_label': cell['type_label'], 'length': cell['length'],
                                       'spatial': [cell['spatial'][0], cell['spatial'][1]],
                                       'spatial_raw': [cell['spatial'][0], cell['spatial'][1]], 'space': 1})
                Xs_right.append(cell['spatial'][0])
            else:
                # 左部的涡虫
                new_Data_left.append({'input_ids': cell['input_ids'], 'label': cell['label'],'type_label': cell['type_label'], 'length': cell['length'],
                                      'spatial': [cell['spatial'][0], cell['spatial'][1]],
                                      'spatial_raw': [cell['spatial'][0], cell['spatial'][1]], 'space': 0})
                Xs_left.append(cell['spatial'][0])
        X_mean_left = np.mean(Xs_left)  # 计算左
        X_std_left = np.std(Xs_left)
        X_mean_right = np.mean(Xs_right)  # 计算右
        X_std_right = np.std(Xs_right)
        epsilon = 1e-8
        for cell in new_Data_left:
            cell['spatial'][0] = (cell['spatial'][0] - X_mean_left) / (X_std_left + epsilon)
        for cell in new_Data_right:
            cell['spatial'][0] = (cell['spatial'][0] - X_mean_right) / (X_std_right + epsilon)
        new_Data = new_Data_left + new_Data_right
        cell_dataset = datasets.Dataset.from_list(new_Data)
        # 计算新的数据集
        Xs = []
        for cell in cell_dataset:
            Xs.append(cell['spatial'][0])
            Ys.append(cell['spatial'][1])
        X_mean = np.mean(Xs)  # 计算平均值
        Y_mean = np.mean(Ys)  # 计算平均值
        Y_std = np.std(Ys)  # 计算标准差
        X_std = np.std(Xs)
        # 避免除以零
        epsilon = 1e-8
        def spatial_aug_fc(spatial_info, noise_scale=0.01):
            noise = np.random.uniform(low=-noise_scale, high=noise_scale, size=spatial_info.shape)
            augmented_spatial_info = spatial_info + noise
            return augmented_spatial_info

        def normalized_spatial(example):
            x = example['spatial'][0]
            y = example['spatial'][1]
            normalized_x = (x - X_mean) / (X_std + epsilon)
            normalized_y = (y - Y_mean) / (Y_std + epsilon)
            example['spatial'] = [normalized_x, normalized_y]
            return example
            pass

        cell_dataset = cell_dataset.map(normalized_spatial, num_proc=16)
        cell_dataset = cell_dataset.shuffle(123)
        train_dataset = cell_dataset.select([i for i in range(0, round(len(cell_dataset) * 0.8))])
        test_dataset = cell_dataset.select([i for i in range(round(len(cell_dataset) * 0.8),round(len(cell_dataset)))])
        p_cell_aug = []
        for cell in train_dataset:
            if(aug):
                aug_number = max(1, int(len(cell['input_ids']) / 200))
                for _ in range(aug_number):
                    num_mask = random.randint(0, int(len(cell['input_ids']) * 0.35))
                    num_genes = min(2048, max(200, len(cell['input_ids']) - num_mask))  # 最多2048，最少200个，如果不足，就不删减
                    random.shuffle(cell['input_ids'])
                    spatial_aug = spatial_aug_fc(np.array(cell['spatial']))
                    p_cell_aug.append({'label': cell['label'],'type_label':cell['type_label'], 'input_ids': cell['input_ids'][:num_genes],
                                       'spatial': np.array(spatial_aug), 'length': len(cell['input_ids'][:num_genes])})
            p_cell_aug.append(cell)
        train_dataset = datasets.Dataset.from_list(p_cell_aug)
        train_dataset = train_dataset.shuffle()
        test_dataset = test_dataset.shuffle()
        train_dataset_all.append(train_dataset)
        test_dataset_all.append(test_dataset)
    # 先通过原始geneformer完成时间信息的嵌入
    train_dataset_all = concatenate_datasets(train_dataset_all)
    test_dataset_all = concatenate_datasets(test_dataset_all)
    train_dataset_all.save_to_disk(output_path+'train_data.dataset')
    test_dataset_all.save_to_disk(output_path+'test_data.dataset')
    train_dataset_all = datasets.Dataset.load_from_disk(output_path+'train_data.dataset')
    test_dataset_all = datasets.Dataset.load_from_disk(output_path+'test_data.dataset')
    if(model_type == 'Bert'):
        time_model = BertForSequenceClassification.from_pretrained(pretrained_path,
                                                                   num_labels=len(timepoints),
                                                                   output_attentions=False,
                                                                   output_hidden_states=False,
                                                                   ignore_mismatched_sizes=True).to('cuda')
    elif(model_type == 'DeBerta'):
        time_model = DebertaForSequenceClassification.from_pretrained(pretrained_path,
                                                                   num_labels=len(timepoints),
                                                                   output_attentions=False,
                                                                   output_hidden_states=False,
                                                                   ignore_mismatched_sizes=True).to('cuda')
    output_dir = output_path
    os.makedirs(output_dir, exist_ok=True)
    geneformer_batch_size = 6
    training_args = {
        "learning_rate": 1e-5,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": round(len(train_dataset_all) / geneformer_batch_size / 10),
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": "linear",
        "warmup_steps": 500,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs":10,
        "load_best_model_at_end": True,
        "output_dir": output_dir,

    }
    training_args_init = TrainingArguments(**training_args)
    # create the trainer
    trainer = Trainer(
        model=time_model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=train_dataset_all,
        eval_dataset=test_dataset_all,
        compute_metrics=compute_metrics
    )
    # train the cell type classifier
    trainer.train()
    trainer.save_model()
    time_pretrained_path = getfolder(output_path) # 保存的最后的模型
    # 各个时期分别判断细胞种类
    for timepoint in timepoints:
        output_dir = output_path + timepoint+'/'
        os.makedirs(output_dir,exist_ok=True)
        train_dataset = train_dataset_all.filter(lambda x: x['label'] == timepoint_label[timepoint])
        test_dataset = test_dataset_all.filter(lambda x: x['label'] == timepoint_label[timepoint])
        train_dataset = train_dataset.shuffle()
        test_dataset = test_dataset.shuffle()
        if(model_type == 'Bert'):
            spatial_model = BertWithSpatialInfo_spatial3.from_pretrained(time_pretrained_path,
                                                                         num_labels=len(celltype_label),
                                                                         output_attentions=False,
                                                                         output_hidden_states=False,
                                                                         ignore_mismatched_sizes=True).to('cuda')
        elif(model_type == 'DeBerta'):
            spatial_model = BertWithSpatialInfo_spatial2.from_pretrained(time_pretrained_path,
                                                                         num_labels=len(celltype_label),
                                                                         output_attentions=False,
                                                                         output_hidden_states=False,
                                                                         ignore_mismatched_sizes=True).to('cuda')
        def collate_fn(batch):
            input_ids = []
            lengths = []
            spatial = []
            for item in batch:
                random.shuffle(item['input_ids'])
                input_ids.append(torch.tensor(item['input_ids'][:2048]))
                lengths.append([item['length']])
                spatial.append(torch.tensor(item['spatial']))
            lengths = torch.tensor(lengths)
            labels = torch.tensor([item['type_label'] for item in batch])
            spatial = torch.stack(spatial)  # 转换为tensor
            # 对 input_ids 进行填充，使长度一致
            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
            return input_ids_padded, labels, spatial, lengths
        train_loader = DataLoader(train_dataset, batch_size=6, collate_fn=collate_fn, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=6, collate_fn=collate_fn, shuffle=True)
        # 定义优化器和损失函数
        optimizer = optim.AdamW(spatial_model.parameters(), lr=3e-5, weight_decay=1e-4)  # weight_decay参数控制L2正则化
        criterion = torch.nn.CrossEntropyLoss()
        # 设置学习率调度器
        num_epochs = 40
        num_training_steps = num_epochs * len(train_loader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps
        )
        best_accuracy = 0.0  # 最佳准确率
        accuracy_list=[]
        f1_list =[]
        for epoch in range(num_epochs):
            # 训练模型
            spatial_model.train()
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0
            progress_bar = tqdm(train_loader)
            correct = 0
            total = 0
            all_labels = []
            all_predictions = []
            for batch in progress_bar:
                input_ids, labels, spatial,lengths = [item.to('cuda') for item in batch]
                total += labels.size(0)
                # 前向传播
                input_ids = input_ids.long()
                # 注意力掩码，确保模型忽略填充的部分
                attention_mask = (input_ids != 0).long()
                spatial = spatial.float()
                # 前向传播，加入空间信息
                outputs,_,_,l2_loss = spatial_model(epoch,input_ids=input_ids, spatial_info=spatial, attention_mask=attention_mask,gene_length = lengths)

                loss = criterion(outputs, labels)
                loss = loss +l2_loss*0.03
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
            print(f"Train Epoch Loss: {epoch_loss / len(train_loader):.4f}, Train Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
            # 测试模型
            spatial_model.eval()
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
                    outputs,_,_,_ = spatial_model(epoch,input_ids=input_ids, spatial_info=spatial, attention_mask=attention_mask,gene_length = lengths)
                    # outputs = spatial_model(input_ids=input_ids, spatial_info=spatial)
                    # print(outputs)
                    _, predicted = torch.max(outputs, 1)

                    # 统计正确的预测
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # 收集所有标签和预测
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
            accuracy = correct / total

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
                torch.save(spatial_model,output_path+timepoint+ "/bert_with_spatial_model_best.pth")

            if((epoch+1)%10 == 0):
                # 每十个epoch保存模型
                torch.save(spatial_model,output_path+timepoint+ "/bert_with_spatial_model_"+str(epoch+1)+".pth")
                print(output_path+timepoint+ "/bert_with_spatial_model_"+str(epoch+1)+".pth saved")

        with open(output_path+timepoint+ "/history.txt",'w')as file:
            for i in range(len(accuracy_list)):
                file.write(str(accuracy_list[i]))
                file.write(' ')
                file.write(str(f1_list[i]))
                file.write('\n')

    pass
