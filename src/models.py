import pickle
import random

import torch
# from dgl.nn.pytorch import GraphConv
from sklearn.decomposition import PCA
from torch import nn, optim
from torch.nn import Linear
from torch.utils.data import TensorDataset, Subset, DataLoader
from tqdm import tqdm
from transformers import BertForSequenceClassification,DebertaForSequenceClassification

class BertWithSpatialInfo_spatial2(DebertaForSequenceClassification):
    # 尝试新的处理方法，不额外输入细胞的全局信息而是提前对数据集做标准化处理
    def __init__(self, config, spatial_dim=2):
        super(BertWithSpatialInfo_spatial2, self).__init__(config)

        self.spatial_fc = nn.Linear(spatial_dim,16)
        self.gene_fc = nn.Linear(config.hidden_size, 16 + config.hidden_size)
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.05)
        # 添加一个MultiheadAttention层
        # 注意力头数：num_heads，隐藏维度：hidden_size
        self.attn = nn.MultiheadAttention(embed_dim=config.hidden_size + 16, num_heads=8, dropout=0.1)
        self.attn2 = nn.MultiheadAttention(embed_dim=config.hidden_size + 16, num_heads=8, dropout=0.1)
        self.attn3 = nn.MultiheadAttention(embed_dim=config.hidden_size + 16, num_heads=8, dropout=0.1)
        # 层标准化
        self.layer_norm_gene = nn.LayerNorm(config.hidden_size + 16)
        self.layer_norm_spatial = nn.LayerNorm(16)
        self.layer_norm_all = nn.LayerNorm(config.hidden_size + 16)
        # 分类器层
        # self.classifier = nn.Linear(config.hidden_size+17, config.num_labels)
        self.classifier  = nn.Sequential(
            nn.Linear(config.hidden_size+17, 256),
            nn.ReLU(),
            nn.Linear(256, config.num_labels)  # ['n_class']
        )
        # 权重参数
        self.alpha = nn.Parameter(torch.clamp(torch.tensor(0.5), 0, 1))


    def unfreeze_layers(self, epoch,padding_epoch):
        """
        根据当前训练的 epoch，从最后一层逐步解冻更多的层。
        """
        num_layers = len(self.deberta.encoder.layer)  # BERT 的总层数
        if(epoch >= padding_epoch):
        # 计算需要解冻的层数，确保不会超过总层数
            layers_to_unfreeze = min(int((epoch + 1 - padding_epoch) * (num_layers / 20)), num_layers)
            # 从最后一层开始解冻
            for i in range(num_layers - layers_to_unfreeze, num_layers):
                for param in self.deberta.encoder.layer[i].parameters():
                    if(param.requires_grad == False):
                        param.requires_grad = True
    def forward(self,epoch,padding_epoch=5, input_ids=None, spatial_info=None, attention_mask=None, gene_length=None, **kwargs):
        # 冻结BERT参数
        # 10 46
        # 10 45 差不多
        # 10 45 差不多
        for param in self.deberta.parameters():
            param.requires_grad = False
        self.unfreeze_layers(epoch,padding_epoch) # 逐步解禁
        # 通过原始BERT模型获得文本嵌入
        outputs = self.deberta(input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = outputs.last_hidden_state[:, 0]
        gene_fc_out = self.gene_fc(pooled_output)
        spatial_output = self.spatial_fc(spatial_info)

        gene_output2 = self.layer_norm_gene(gene_fc_out)
        spatial_output2 = self.layer_norm_spatial(spatial_output)
        # 拼接空间信息和基因信息
        combined_embed = torch.cat([pooled_output,spatial_output2], dim=-1)  # shape: [batch_size, seq_len, hidden_size + 6]
        combined_embed = combined_embed + gene_fc_out
        combined_embed = self.layer_norm_all(combined_embed)
        # 将combined_embed传入multihead attention
        # 注意：需要对输入进行调整，符合 [seq_len, batch_size, embed_dim] 的格式
        combined_embed2 = combined_embed.unsqueeze(1)
        combined_embed2 = combined_embed2.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size + 6]
        gene_output2 = gene_output2.unsqueeze(1)
        gene_output2 = gene_output2.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size + 6]
        # 使用multihead attention层进行处理
        attn_output1, _ = self.attn(combined_embed2, gene_output2, gene_output2)
        attn_output2, _ = self.attn2(gene_output2,combined_embed2, combined_embed2)
        attn_output3, _ = self.attn3(combined_embed2,gene_output2,  combined_embed2)
        attn_output = attn_output1 + attn_output2 + attn_output3
        # 反转回原始的batch_size, seq_len, embed_dim
        attn_output = attn_output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size + 6]
        # 添加Dropout
        attn_output = self.dropout(attn_output)
        # 获取每个token的最终表示
        attn_output = attn_output.mean(dim=1)  # [batch_size, hidden_size + 6]
        attn_output = self.layer_norm_all(attn_output)
        attn_output = attn_output*(1-self.alpha) + combined_embed *self.alpha
        #pooled_output = self.layer_norm_all(pooled_output)
        combined_output = torch.cat((attn_output,gene_length), dim=1)
        # 通过dropout，避免过拟合
        combined_output = self.dropout(combined_output)
        # 通过分类器获得最终输出
        logits = self.classifier(combined_output )
        l2_reg = sum(torch.norm(param, p=2) for param in self.parameters()).detach()

        return logits, combined_output,attn_output,l2_reg  # 返回不同层的输出

class BertWithSpatialInfo_spatial3(BertForSequenceClassification):
    # 尝试新的处理方法，不额外输入细胞的全局信息而是提前对数据集做标准化处理
    def __init__(self, config, spatial_dim=2):
        super(BertWithSpatialInfo_spatial3, self).__init__(config)

        self.spatial_fc = nn.Linear(spatial_dim,16)
        self.gene_fc = nn.Linear(config.hidden_size, 16 + config.hidden_size)
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.05)
        # 添加一个MultiheadAttention层
        # 注意力头数：num_heads，隐藏维度：hidden_size
        self.attn = nn.MultiheadAttention(embed_dim=config.hidden_size + 16, num_heads=8, dropout=0.1)
        self.attn2 = nn.MultiheadAttention(embed_dim=config.hidden_size + 16, num_heads=8, dropout=0.1)
        self.attn3 = nn.MultiheadAttention(embed_dim=config.hidden_size + 16, num_heads=8, dropout=0.1)
        # 层标准化
        self.layer_norm_gene = nn.LayerNorm(config.hidden_size + 16)
        self.layer_norm_spatial = nn.LayerNorm(16)
        self.layer_norm_all = nn.LayerNorm(config.hidden_size + 16)
        # 分类器层
        # self.classifier = nn.Linear(config.hidden_size+17, config.num_labels)
        self.classifier  = nn.Sequential(
            nn.Linear(config.hidden_size+17, 256),
            nn.ReLU(),
            nn.Linear(256, config.num_labels)  # ['n_class']
        )
        # 权重参数
        self.alpha = nn.Parameter(torch.clamp(torch.tensor(0.5), 0, 1))

    def unfreeze_layers(self, epoch,padding_epoch):
        """
        根据当前训练的 epoch，从最后一层逐步解冻更多的层。
        """
        num_layers = len(self.bert.encoder.layer)  # BERT 的总层数
        if(epoch >= padding_epoch):
        # 计算需要解冻的层数，确保不会超过总层数
            layers_to_unfreeze = min(int((epoch + 1 - padding_epoch) * (num_layers / 20)), num_layers)
            # 从最后一层开始解冻
            for i in range(num_layers - layers_to_unfreeze, num_layers):
                for param in self.bert.encoder.layer[i].parameters():
                    if(param.requires_grad == False):
                        param.requires_grad = True
    def forward(self,epoch,padding_epoch=5, input_ids=None, spatial_info=None, attention_mask=None, gene_length=None, **kwargs):
        # 冻结BERT参数
        for param in self.bert.parameters():
            param.requires_grad = False
        self.unfreeze_layers(epoch,padding_epoch) # 逐步解禁
        # 通过原始BERT模型获得文本嵌入
        outputs = self.bert(input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = outputs.pooler_output  # BERT的池化输出
        gene_fc_out = self.gene_fc(pooled_output)
        spatial_output = self.spatial_fc(spatial_info)

        gene_output2 = self.layer_norm_gene(gene_fc_out)
        spatial_output2 = self.layer_norm_spatial(spatial_output)
        # 拼接空间信息和基因信息
        combined_embed = torch.cat([pooled_output,spatial_output2], dim=-1)  # shape: [batch_size, seq_len, hidden_size + 6]
        combined_embed = combined_embed + gene_fc_out
        combined_embed = self.layer_norm_all(combined_embed)
        # 将combined_embed传入multihead attention
        # 注意：需要对输入进行调整，符合 [seq_len, batch_size, embed_dim] 的格式
        combined_embed2 = combined_embed.unsqueeze(1)
        combined_embed2 = combined_embed2.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size + 6]
        gene_output2 = gene_output2.unsqueeze(1)
        gene_output2 = gene_output2.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size + 6]
        # 使用multihead attention层进行处理
        attn_output1, _ = self.attn(combined_embed2, gene_output2, gene_output2)
        attn_output2, _ = self.attn2(gene_output2,combined_embed2, combined_embed2)
        attn_output3, _ = self.attn3(combined_embed2,gene_output2,  combined_embed2)
        attn_output = attn_output1 + attn_output2 + attn_output3
        # 反转回原始的batch_size, seq_len, embed_dim
        attn_output = attn_output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size + 6]
        # 添加Dropout
        attn_output = self.dropout(attn_output)
        # 获取每个token的最终表示
        attn_output = attn_output.mean(dim=1)  # [batch_size, hidden_size + 6]
        attn_output = self.layer_norm_all(attn_output)
        attn_output = attn_output*(1-self.alpha) + combined_embed *self.alpha
        #pooled_output = self.layer_norm_all(pooled_output)
        combined_output = torch.cat((attn_output,gene_length), dim=1)
        # 通过dropout，避免过拟合
        combined_output = self.dropout(combined_output)
        # 通过分类器获得最终输出
        logits = self.classifier(combined_output )
        l2_reg = sum(torch.norm(param, p=2) for param in self.parameters()).detach()

        return logits, combined_output,attn_output,l2_reg  # 返回不同层的输出

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


import pickle
import random
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.nn import Linear
# import dgl
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                    }, f'./models/MLP_model2/mlp_model_checkpoint_epoch_{epoch + 1}.pth')

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