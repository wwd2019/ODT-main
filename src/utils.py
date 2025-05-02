import gzip
import random
import sys

import math
from torch.nn.utils.rnn import pad_sequence
import anndata
import datasets
import torch
from sklearn.metrics import accuracy_score, f1_score
import scanpy as sc
import os,re
import pickle
from copy import copy,deepcopy
import pandas as pd
import numpy as np
import statistics
import time
from tqdm import trange, tqdm
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA, FileTransferSpeed

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
def gzip_file(input_path, output_path):
    # 将对应文件原地解压缩
    with gzip.open(input_path, 'rb') as f_in:
        if(os.path.exists(output_path)):
                print(output_path+' 已经存在')
        else:
            with open(output_path, 'wb') as f_out:
                try:
                    f_out.write(f_in.read())
                except:
                    print('err '+input_path)
    print(input_path+' unzip success')
def traverse_files(root_path):
    # 遍历路径下所有的.gz文件
    file_paths=[]
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if(file.endswith('.gz')):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return  file_paths
def gzip_all(root_path,target_path = './dataSets/dataSets_filtered_unzipped/'):
    gzfiles=traverse_files(root_path)
    print(len(gzfiles))
    for gzfile in gzfiles:
        new_path=gzfile.replace('.gz', '')
        new_path=new_path.split('/')
        new_path=new_path[-1]
        new_file_path=target_path + new_path
        gzip_file(gzfile,new_file_path)
    pass
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
def makedata_labelled(input_path,output_path='./data_treated_sets/labelled_p_cell_with_label.pkl'):
    data_all=[]
    widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ', FileTransferSpeed()]
    progress = ProgressBar(widgets=widgets)
    sc.set_figure_params(facecolor="white", figsize=(8, 8), dpi=100, color_map='viridis_r')
    sc.settings.verbosity = 3  # 设置日志等级: errors (0), warnings (1), info (2), hints (3)
    sc.logging.print_header()
    print(os.getcwd())  # 查看当前路径
    # os.chdir('./filtered_gene_bc_matrices/scanpy') #修改路径
    datdir = input_path
    # 导入 stereo-Seq 数据
    adata = sc.read_h5ad(datdir)
    adata.var_names_make_unique()  # 索引去重，若上一步中使用 `var_names='gene_ids'` 则这一步非必须进行
    # 用于存储分析结果文件的路径
    results_file = output_path
    # 基础过滤：去除表达基因200以下的细胞；去除在3个细胞以下表达的基因。
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    mat = pd.DataFrame(data=adata.X.todense(), index=adata.obs_names.values, columns=adata.var_names.values)
    # mat.to_csv(os.path.join(outdir, 'mat.csv'), index=True)
    spatial = adata.obsm['spatial'].join(pd.DataFrame(adata.obs[['orig.ident', 'timepoint', 'unified_domain_5']]))
    rep = ['1', '3', '5']
    reps = '|'.join(rep)
    spatial_rep1 = spatial[spatial['orig.ident'].str.contains(reps)]
    mat_rep1 = mat.loc[spatial['orig.ident'].str.contains(reps)]
    for slice_ in np.unique(spatial_rep1['timepoint']):
        print(slice_)
        spatial_rep1_slice = spatial_rep1.loc[spatial_rep1['timepoint'] == slice_, :].sort_index()
        mat_rep1_slice = mat_rep1.loc[spatial_rep1_slice.index, :].sort_index()
        print(mat_rep1_slice.shape)
    cell_names = adata.obs_names.values
    print(spatial.shape)
    print(len(cell_names))
    datalists = mat.values
    data_dict = {}
    col = mat.columns
    for i in tqdm(range(len(cell_names))):
        item = cell_names[i]
        data_dict[item] = []
        for j in range(len(col)):
            if (datalists[i][j] != 0):
                data_dict[item].append(col[j] + ':' + str(datalists[i][j]))
    i=0
    print(len(data_dict))
    j=0
    for cell in data_dict.keys():
        if(spatial['orig.ident'][i] and spatial['timepoint'][i] and spatial['unified_domain_5'][i]):
            data_all.append({'cell_genes': data_dict[cell],'length': len(data_dict[cell]), 'indent':spatial['orig.ident'][i],'timepoint':spatial['timepoint'][i],'unified_domain':spatial['unified_domain_5'][i]})
        else:
            data_all.append(
                {'cell_genes': data_dict[cell], 'length': len(data_dict[cell]), 'indent': 'unknown','timepoint':'unknown', 'unified_domain': 'unknown'})
            j+=1
        i+=1
    print(j)
    with open(output_path, 'wb') as f1:
        pickle.dump(data_all,f1)

def view_bar(message, num, total):
    # 进度条
    rate = num / total
    rate_num = int(rate * 30)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, "|" * rate_num, " " * (30 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()

def h5_data_make():
    # 将50万无标签细胞数据转成h5ad格式，所有细胞的表达量默认1
    # 加载所有可用的smesg基因id
    smesg_all_list = []
    with open("./tk_dict/gene_tk.pkl", "rb") as fp:
        token_dictionary = pickle.load(fp)
    smesg_all_list = list(token_dictionary.keys())[:-2]

    with open('./data_treated_sets/other_valid_p_cell/p_cell_unannotated.pkl', 'rb') as f:
        p_unlabelled_cells = pickle.load(f)
    # 构建基因表达矩阵
    num_cells = len(p_unlabelled_cells)
    num_genes = len(smesg_all_list)

    # 细胞 ID 列表
    cell_ids = [cell['cell_id'] for cell in p_unlabelled_cells]

    # 创建空矩阵
    expression_matrix = np.zeros((num_cells, num_genes), dtype=np.int8)

    # 填充表达矩阵（基因表达量设为 1）
    gene_index_map = {gene: i for i, gene in enumerate(smesg_all_list)}

    for cell_idx, cell in tqdm(enumerate(p_unlabelled_cells)):
        for gene in cell['cell_genes']:
            if gene in gene_index_map:
                expression_matrix[cell_idx, gene_index_map[gene]] = 1

    # 转换为 AnnData 对象
    adata = anndata.AnnData(X=expression_matrix, dtype=np.int8)
    adata.var_names = smesg_all_list  # 基因名称
    adata.obs_names = cell_ids  # 细胞 ID

    # 保存为 h5ad 格式
    adata.write("./data_treated_sets/other_valid_p_cell/unlabelled_cells.h5ad")
    cell_gene_counts = (adata.X > 0).sum(axis=1)
    adata = adata[cell_gene_counts >= 50].copy()
    adata.write("./data_treated_sets/other_valid_p_cell/unlabelled_cells_over50.h5ad")
    print("转换完成，已保存为 'unlabelled_cells.h5ad'")
    pass
def model_get(gene_model_path,num_labels):
# 1. 先加载基因模型（gene_model）
    gene_model = torch.load(gene_model_path)
    # 2. 初始化 spatial_model（它是基于 gene_model 扩展的）
    config = gene_model.config
    config.num_labels = num_labels
    spatial_model = BertWithSpatialInfo_spatial3(config).to('cuda')
    # 获取 gene_model 的权重
    gene_model_dict = gene_model.state_dict()
    # 获取 spatial_model 的当前权重
    spatial_model_dict = spatial_model.state_dict()
    # 过滤掉 classifier 层的参数（避免维度不匹配）
    filtered_gene_model_dict = {k: v for k, v in gene_model_dict.items() if k in spatial_model_dict and v.shape == spatial_model_dict[k].shape}
    # 加载匹配的权重
    spatial_model_dict.update(filtered_gene_model_dict)
    spatial_model.load_state_dict(spatial_model_dict)
    print("Successfully loaded pre-trained weights (excluding classifier).")
    return spatial_model