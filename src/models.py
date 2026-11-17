import pickle
import random

import torch
# from dgl.nn.pytorch import GraphConv
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn, optim
from torch.nn import Linear
from torch.utils.data import TensorDataset, Subset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import BertForSequenceClassification, DebertaForSequenceClassification, BertModel, Trainer

import pickle
import random
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.nn import Linear
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64, dropout_rate=0.01):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Dropout2d(dropout_rate),  # Add dropout after ReLU
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128*38*38, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*38*38),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 38, 38)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout2d(dropout_rate),  # Add dropout after ReLU
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, return_latent=False):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        # 👇裁剪回和输入一致
        recon = recon[:, :, :x.shape[2], :x.shape[3]]
        if return_latent:
            return latent
        else:
            return recon

class GeneSnapshotDataset(Dataset):

    def __init__(self, preprocessed_snapshots):
        self.genes = list(preprocessed_snapshots.keys())
        self.snapshots = [
            torch.tensor(preprocessed_snapshots[g], dtype=torch.float32).unsqueeze(0)
            for g in self.genes
        ]

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, idx):
        return self.snapshots[idx], self.genes[idx]

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.eps = eps

        self.scale = nn.Parameter(
            torch.ones(dim)
        )

    def forward(self, x):

        # x: (..., dim)

        rms = torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdim=True)
            + self.eps
        )

        x = x * rms

        return self.scale * x
class BertWithSpatialInfo_spatial(BertForSequenceClassification):
    def __init__(self, config, spatial_dim=2, freeze=True):
        super().__init__(config)
        # =====================================================
        # backbone
        # =====================================================
        self.bert = BertModel(config)
        hidden = config.hidden_size
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

            for param in self.bert.encoder.layer[-2:].parameters():
                param.requires_grad = True

            for name, p in self.bert.named_parameters():
                if "LayerNorm" in name:
                    p.requires_grad = True

        # =====================================================
        # spatial encoding (lightweight, stable)
        # =====================================================
        self.fourier_dim = 32
        self.freq_scale = nn.Parameter(torch.tensor(1.0))

        self.spatial_proj = nn.Sequential(
            nn.Linear(self.fourier_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden)
        )

        # =====================================================
        # spatial branch
        # =====================================================
        self.spatial_attn = nn.MultiheadAttention(
            hidden, 2, batch_first=True
        )

        # =====================================================
        # fusion gate (CRITICAL)
        # =====================================================
        self.gate_g = nn.Linear(hidden, hidden)
        self.gate_s = nn.Linear(hidden, hidden)
        self.g_alpha = nn.Linear(hidden, 1)
        self.s_alpha = nn.Linear(hidden, 1)

        self.length_proj = nn.Linear(1, hidden)
        # =====================================================
        # final fusion MLP
        # =====================================================
        self.fusion = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden)
        )

        # =====================================================
        # heads
        # =====================================================
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, config.num_labels)
        )

        self.spatial_pred = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, spatial_dim)
        )
        self._init_new_weights()

    def _init_new_weights(self):

        modules = [
            self.spatial_proj,
            self.spatial_attn,
            self.gate_g,
            self.gate_s,
            self.g_alpha,
            self.s_alpha,
            self.length_proj,
            self.fusion,
            self.classifier,
            self.spatial_pred
        ]

        for module in modules:

            for m in module.modules():

                if isinstance(m, nn.Linear):

                    nn.init.normal_(
                        m.weight,
                        mean=0.0,
                        std=0.02
                    )

                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


                elif isinstance(m, nn.LayerNorm):

                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)


                elif isinstance(m, nn.MultiheadAttention):

                    nn.init.normal_(
                        m.in_proj_weight,
                        mean=0.0,
                        std=0.02
                    )

                    if m.in_proj_bias is not None:
                        nn.init.zeros_(
                            m.in_proj_bias
                        )

                    nn.init.normal_(
                        m.out_proj.weight,
                        mean=0.0,
                        std=0.02
                    )

                    if m.out_proj.bias is not None:
                        nn.init.zeros_(
                            m.out_proj.bias
                        )

        nn.init.constant_(
            self.freq_scale,
            1.0
        )
    # =====================================================
    # spatial encoding
    # =====================================================
    def build_spatial_encoding(self, input_ids):
        token_ids = input_ids.float()
        half = self.fourier_dim // 2

        freq = torch.arange(half, device=input_ids.device).float()
        freq = torch.exp(-torch.log(torch.tensor(10000.0)) * (freq / half))
        freq = freq * torch.exp(self.freq_scale)

        angles = token_ids.unsqueeze(-1) * freq
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    # =====================================================
    # forward
    # =====================================================
    def forward(
            self,
            input_ids,
            attention_mask=None,
            labels=None,
            lengths=None,
            spatials=None
    ):

        # =====================================================
        # 1. BERT encoding
        # =====================================================
        x = self.bert(
            input_ids,
            attention_mask=attention_mask
        ).last_hidden_state  # [B, L, H]
        # =====================================================
        # 2. spatial encoding
        # =====================================================
        spatial = self.build_spatial_encoding(input_ids)
        spatial = self.spatial_proj(spatial)
        # =====================================================
        # 3. dynamic spatial query
        # =====================================================
        # 用 gene context 动态查询 spatial prior
        mask = attention_mask.unsqueeze(-1).float()
        # gene queries spatial context
        s, _ = self.spatial_attn(
            x,
            spatial,
            x
        )
        # broadcast 到 token level
        s = s.expand(-1, x.size(1), -1)

        # =====================================================
        # Step 4 -- Softmax-gated fusion
        # =====================================================
        g_alpha = self.g_alpha(x)  # [B, L, 1]
        s_alpha = self.s_alpha(s)  # [B, L, 1]

        alpha = torch.cat([g_alpha, s_alpha], dim=-1)
        alpha = F.softmax(alpha, dim=-1)

        g_weight = alpha[..., 0:1]
        s_weight = alpha[..., 1:2]

        h = x * g_weight + s * s_weight

        # =====================================================
        # 6. masked pooling
        # =====================================================
        pooled = (
                (h * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1e-6)
        )

        # =====================================================
        # 7. length embedding
        # =====================================================
        lengths = torch.log1p(lengths.float())
        lengths = lengths.unsqueeze(-1)

        gene_output = pooled + self.length_proj(lengths)

        # =====================================================
        # 8. heads
        # =====================================================
        logits = self.classifier(gene_output)

        spatial_pred = self.spatial_pred(gene_output)

        # =====================================================
        # inference
        # =====================================================
        if labels is None:
            return (None, logits, gene_output)

        # =====================================================
        # 9. classification loss
        # =====================================================
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        # =====================================================
        # 10. spatial regression loss
        # =====================================================
        if spatials is not None:
            spatial_loss = F.smooth_l1_loss(
                spatial_pred,
                spatials
            )
            # 平滑上限
            spatial_loss = torch.tanh(spatial_loss / 3.0)
            loss = loss + 0.1 * spatial_loss

        return (loss, logits, gene_output)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))

        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))

        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Accuracy
    acc = accuracy_score(labels, preds)

    # Precision
    precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    precision_micro = precision_score(labels, preds, average='micro', zero_division=0)
    precision_weighted = precision_score(labels, preds, average='weighted', zero_division=0)
    # Recall
    recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
    recall_micro = recall_score(labels, preds, average='micro', zero_division=0)
    recall_weighted = recall_score(labels, preds, average='weighted', zero_division=0)
    # F1
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "precision_micro": precision_micro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_micro": recall_micro,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }

def get_weighted_sampler(
        dataset,
        label_key="label",
        multiplier=1.3,
        alpha=0.7,
        min_weight=0.8,
        max_weight=2.5,
        eps=1e-8
):
    labels = np.array([int(x[label_key]) for x in dataset])
    classes, counts = np.unique(labels, return_counts=True)

    imbalance_ratio = counts.max() / (counts.min() + eps)

    print(f"class counts = {counts.tolist()}")
    print(f"imbalance ratio = {imbalance_ratio:.2f}")

    # inverse frequency weighting
    freq = counts / counts.sum()
    class_weights = 1.0 / (freq + eps)
    class_weights /= class_weights.mean()

    # soften weighting
    class_weights = class_weights ** alpha
    class_weights /= class_weights.mean()

    # limit extreme re-weighting
    class_weights = np.clip(
        class_weights,
        min_weight,
        max_weight
    )

    class_weights /= class_weights.mean()

    print(
        "final class weights:",
        {
            int(c): round(float(w), 3)
            for c, w in zip(classes, class_weights)
        }
    )

    weight_dict = dict(zip(classes, class_weights))

    sample_weights = np.array(
        [weight_dict[l] for l in labels],
        dtype=np.float32
    )

    sample_weights /= sample_weights.mean()

    return WeightedRandomSampler(
        weights=torch.tensor(
            sample_weights,
            dtype=torch.float32
        ),
        num_samples=int(len(dataset) * multiplier),
        replacement=True
    )

from transformers import Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW

class WeightedTrainer(Trainer):

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)



        # =====================================================
        # weighted sampler
        # =====================================================
        self.sampler = get_weighted_sampler(
            self.train_dataset
        )

    # =====================================================
    # dataloader
    # =====================================================

    def get_train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self.sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True
        )


class MLPmodel2(nn.Module):
    def __init__(self, model_dict, device):
        super(MLPmodel2, self).__init__()
        self.device = device
        # 定义模型层
        hidden_sizes = model_dict['hidden_sizes']
        # 输入层到第一个隐藏层
        self.fc1 = nn.Linear(model_dict['input_size'], hidden_sizes[0]).to(self.device)
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0]).to(self.device)
        # 中间隐藏层
        self.h1 = nn.Linear(hidden_sizes[0], hidden_sizes[1]).to(self.device)
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1]).to(self.device)
        self.h2 = nn.Linear(hidden_sizes[1], hidden_sizes[2]).to(self.device)
        self.bn3 = nn.BatchNorm1d(hidden_sizes[2]).to(self.device)
        # 残差连接层
        self.fv1 = nn.Linear(model_dict['input_size'], hidden_sizes[1]).to(self.device)
        self.fv2 = nn.Linear(hidden_sizes[0], hidden_sizes[-1]).to(self.device)
        self.fv3 = nn.Linear(hidden_sizes[1], model_dict['num_classes']).to(self.device)
        # 输出层
        self.fc2 = nn.Linear(hidden_sizes[-1], model_dict['num_classes']).to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device)
        # Dropout
        self.dropout = nn.Dropout(model_dict['dropout']).to(self.device)
        # 权重参数
        self.alpha = nn.Parameter(torch.clamp(torch.tensor(0.5), 0, 1)).to(self.device)
        self.beta = nn.Parameter(torch.clamp(torch.tensor(0.5), 0, 1)).to(self.device)
        self.theta = nn.Parameter(torch.clamp(torch.tensor(0.5), 0, 1)).to(self.device)  # 初始值为 0.5，可训练
    def forward(self, x):
        # 输入层到第一个隐藏层
        x1 = self.fv1(x)  # input -> h1
        x = self.fc1(x)  # input -> h0
        x = F.gelu(self.bn1(x)) + x  # 残差连接
        # 隐藏层 1
        x2 = self.fv2(x)  # h0 -> h2
        x = self.h1(x)  # h0 -> h1
        x = F.gelu(self.bn2(x))*(1-self.alpha) + x1*self.alpha  # 残差连接
        x = self.dropout(x)  # dropout
        # 隐藏层 2
        x3 = self.fv3(x1)  # h1 -> output
        x = self.h2(x)  # h1 -> h2
        x = F.gelu(self.bn3(x))*(1-self.beta) + x2*self.beta  # 残差连接
        x = self.dropout(x)  # dropout
        # 输出层
        out = self.fc2(x)*self.theta + x3*(1-self.theta )  # h2 -> output, h1 -> output
        out = self.softmax(out)
        return out

class Graph_predicate():
    def __init__(self,gene_tensor_path,gene_disturb_path,target_edges,device='cpu'):
        # gene_tensor_path,gene_disturb_path 是gene数据文件的存储路径，target_edges是已知的可用来监督训练的标签数据
        # 数据读取
        with open(gene_tensor_path,'rb') as f1:
            self.gene_tensor = pickle.load(f1)
        with open(gene_disturb_path,'rb') as f2:
            self.gene_disturb = pickle.load(f2)
        self.target_edges = target_edges
        self.device=device
        # 完成基因数据准备
        self.gene_names = [gene_name for gene_name in self.gene_tensor.keys() if gene_name in self.gene_disturb.keys()]
        self.gene_dict = {}
        for gene in self.gene_names:
            self.gene_dict[gene] = torch.from_numpy(np.hstack((np.array(self.gene_tensor[gene]),np.array(self.gene_disturb[gene]))))
        # 完成训练数据准备
        edge_train_tensors=[]
        edge_train_labels=[]
        edge_train_names=[]
        for edge in target_edges:
            if(edge[0] in self.gene_names and edge[1] in self.gene_names):
                # 选择调控网络和大模型都有的基因
                edge_train_tensors.append([self.gene_dict[edge[0]],self.gene_dict[edge[1]]])
                edge_train_labels.append(edge[2])
                edge_train_names.append([edge[0],edge[1]])
                # edge_train_tensors.append([self.gene_dict[edge[1]],self.gene_dict[edge[0]]])
                # edge_train_labels.append(edge[2])
                # edge_train_names.append([edge[1],edge[0]])
        self.edge_train_datas = [edge_train_tensors,edge_train_labels,edge_train_names]
        pass
    def load_AEmodel(self,model_path):
        # print(self.edge_train_datas[0][0])
        self.model = AE( n_enc_1=800,
            n_enc_2=1200,
            n_enc_3=2600,
            n_dec_1=2600,
            n_dec_2=1200,
            n_dec_3=800,
            n_input=self.edge_train_datas[0][0][0].shape[0],
            n_z=128)
        self.model.load_state_dict(torch.load(model_path))
        print('AE model loaded')
    def get_Tensor(self):
        self.gene_model_tensors={gene_name:torch.tensor(self.gene_dict[gene_name]) for gene_name in self.gene_dict}
        return self.gene_model_tensors
        pass
    def get_Model_tensor(self):
        self.gene_model_tensors = {}
        gene_tensors=[]
        for gene in self.gene_names:
            gene_tensors.append(self.gene_dict[gene].float())
        _,gene_tensors = self.model(torch.stack(gene_tensors))
        # print(len(gene_tensors))14357
        # print(len(self.gene_names))14357
        for i in range(len(gene_tensors)):
            self.gene_model_tensors[self.gene_names[i]] = gene_tensors[i]
        return self.gene_model_tensors
    def Graph_generator(self,num_epochs=500):
        # 此函数用于生成基因调控网络
        self.edge_dict={} # 此字典通过相关基因名确定边节点的id
        gene_all_names=self.gene_names
        gene_all_names2=gene_all_names
        id=0
        for gene1 in gene_all_names:
            for gene2 in gene_all_names2:
                if(gene1 == gene2 or (gene1,gene2) in self.edge_dict):
                    # 取出自己链接自己和重复边
                    pass
                self.edge_dict[(gene1,gene2)]=id
                id+=1
        print('nodes_len',len(self.edge_dict))
        # 构建图
        sources = []
        targets = []
        # 创建邻接矩阵
        for (gene1,gene2) in self.edge_dict:
            for (gene3,gene4) in self.edge_dict:
                if (gene1 in (gene3,gene4) or gene2 in (gene3,gene4)) and (gene1,gene2) != (gene3,gene4):
                    sources.append(self.edge_dict[(gene1,gene2)])
                    targets.append(self.edge_dict[(gene3,gene4)])
        print(len(sources),len(targets))
        edge_nodes = []
        # 创建结点的特征向量
        for (gene1, gene2) in self.edge_dict:
            edge_nodes.append(self.gene_dict[gene1]+self.gene_dict[gene2])

    def MLPmodel(self,model_dict):
        # self.mlpModel = MLPmodel(model_dict,self.device).to(self.device).float()
        self.mlpModel = MLPmodel2(model_dict, self.device).to(self.device).float()
        # criterion = nn.CrossEntropyLoss().to(self.device)
        # criterion =nn.MSELoss().to(self.device)
        optimizer = optim.Adam(self.mlpModel.parameters(), lr=1e-5, weight_decay=1e-4)
        # for param in self.mlpModel.parameters():
        #     print(param.requires_grad)  # 应该输出 True
        num_epochs = model_dict['num_epochs']
        X_tensor=[]
        for edge in self.edge_train_datas[2]:
            # result = torch.cat((tensor1, tensor2), dim=0)
            X_tensor.append(torch.cat((self.gene_model_tensors[edge[0]],self.gene_model_tensors[edge[1]]),dim=0))
        X_tensor = torch.stack(X_tensor)
        y_tensor = torch.tensor(self.edge_train_datas[1], dtype=torch.double)
        mean = X_tensor.mean(dim=1, keepdim=True)  # 形状为 (n, 1)
        std = X_tensor.std(dim=1, keepdim=True)  # 形状为 (n, 1)
        # 标准化输入
        epsilon = 1e-8  # 避免除以零
        inputs_standardized = (X_tensor - mean) / (std + epsilon)  # 结果形状为 (n, m)

        self.dataset = TensorDataset(inputs_standardized, y_tensor)
        dataset = TensorDataset(inputs_standardized, y_tensor)
        dataset_size = len(dataset)
        batch_size = 64
        # 在训练前随机划分数据集
        indices = list(range(dataset_size))
        random.shuffle(indices)
        # 划分训练集和测试集
        train_size = int(0.8 * dataset_size)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        # 创建训练集和测试集
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        # 假设类别权重
        class_weights = torch.tensor([7.0, 3.0, 1.0, 0.5, 1.0, 3.0, 7.0]).to(self.device)
        class_weights = class_weights / class_weights.sum()
        print(dataset_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        for epoch in tqdm(range(num_epochs)):
            # 训练阶段
            self.mlpModel.train()
            total_samples1 = 0  # 总样本数
            correct_predictions1 = 0  # 正确预测数
            for batch_idx, (inputs, targets) in enumerate(train_loader):

                inputs, targets = inputs.to(self.device).float(), targets.to(self.device).long()
                outputs = self.mlpModel(inputs)
                optimizer.zero_grad()
                out = torch.argmax(outputs, dim=1)
                # 计算差的绝对值
                absolute_difference = torch.abs(out - targets)
                # 计算绝对值之和
                # loss = torch.sum(absolute_difference).float()
                loss = F.cross_entropy(outputs, targets,weight=class_weights)
                # lamda = 0.5
                # loss = loss1 *lamda + loss2 *(1-lamda)
                loss.backward()
                optimizer.step()
                correct_predictions1 += torch.sum(out == targets).item()
                total_samples1 += targets.size(0)

            # 测试阶段
            self.mlpModel.eval()
            with torch.no_grad():
                total_samples2 = 0  # 总样本数
                correct_predictions2 = 0  # 正确预测数
                total_absolute_difference = 0  # 绝对值差总和

                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.device).float(), targets.to(self.device).long()
                    outputs= self.mlpModel(inputs)
                    out = torch.argmax(outputs, dim=1)

                    # 计算差的绝对值
                    absolute_difference = torch.abs(out - targets)
                    total_absolute_difference += torch.sum(absolute_difference).item()

                    # 统计正确预测的数量
                    correct_predictions2 += torch.sum(out == targets).item()
                    total_samples2 += targets.size(0)

                # 计算绝对值差的平均值
                test_loss = total_absolute_difference / total_samples2

                # 计算正确率
                accuracy2 = correct_predictions2/ total_samples2 * 100  # 百分比表示
                accuracy1 = correct_predictions1 / total_samples1 * 100  # 百分比表示
            if (epoch+1)%100 ==0:
                print(
                f'Epoch [{epoch + 1}/{num_epochs}],Train Accuracy: {accuracy1:.2f}%, Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f},Test Accuracy: {accuracy2:.2f}%')

            # print('test_loss',test_loss)
            # 保存模型
            if (epoch + 1) % 100 == 0 :
                if (epoch + 1) % 1000 == 0:
                    print(f'Saving checkpoint at epoch {epoch + 1}')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.mlpModel.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, f'/home/wwd/codebox/ODT-main-final/models/MLP_model2/mlp_model_checkpoint_epoch_{epoch + 1}.pth')

                if (epoch + 1) % 500 == 0: # 每500个输出检查一下

                    for i in range(min(20, outputs.shape[0])):
                        probabilities = F.softmax(outputs[i], dim=0)  # Softmax along the class dimension
                        # Get the predicted class
                        predicted_class = torch.argmax(probabilities).item()  # Get the class with the highest probability

                        # Get the true class from targets
                        true_class = targets[i].item()  # Assuming targets are already in scalar form

                        # Print the results
                        print(f'id{i} Prediction Probabilities: {probabilities.tolist()}')  # Convert to list for readable output
                        print(f'Predicted Class: {predicted_class}, Target: {true_class}')
            pass
    def loadMLPmodel(self,model_path,model_dict):
        # 加载模型检查点
        checkpoint = torch.load(model_path)
        self.mlpModel= MLPmodel2(model_dict,self.device).to(self.device).double()
        # 恢复模型的状态
        self.mlpModel.load_state_dict(checkpoint['model_state_dict'])
        print('MLPmodel load success')
        # 如果你想继续训练模型，确保模型处于训练模式
        self.mlpModel.train()
    def model_prdicate(self,model_name,x):
        result_list=[]
        if(model_name=='MLP_model'):
            outputs = self.mlpModel(x) # 直接使用就不需要l2了
            # 1. 使用 Softmax 计算每个类别的概率分布
            probabilities = F.softmax(outputs, dim=1)  # 在类别维度上做softmax，dim=1

            return probabilities

        if(model_name=='AE_model'):
            result_list=self.model(x)
        return result_list