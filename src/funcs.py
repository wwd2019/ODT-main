import pickle
import random
import sys
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datasets
import numpy as np
import scanpy as sc
from tqdm import tqdm
CODE_PATH = "/home/wwd/codebox/ODT-main-final/"
if CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)
def h5ad2datasets(data_path = '/home/wwd/codebox/ODT-main-final/dataSets/spatial_unified_domain_50.h5ad'):
    # 将h5ad 转化成datasets
    adata = sc.read(data_path)
    # print(adata)
    # 使用 adata.var_names 作为基因名列表
    gene_names = np.array(adata.var_names)
    gene_names = np.array([gene_name.split('-')[0] for gene_name in gene_names])
    # print(gene_names[:10])
    # 如果 X 是稀疏矩阵，转为 CSR 格式以按行高效访问
    X = adata.X.tocsr() if hasattr(adata.X, 'tocsr') else adata.X
    cell_list = []
    for i in tqdm(range(X.shape[0])):
        cell_id = adata.obs_names[i]
        celltype = adata.obs.loc[cell_id, 'celltype']
        timepoint = adata.obs.loc[cell_id, 'timepoint']
        expr = X[i].toarray().flatten() if hasattr(X[i], 'toarray') else np.array(X[i]).flatten()
        # 获取非零表达的索引
        nonzero_idx = np.where(expr > 0)[0]
        # 如果该细胞有非零表达
        if len(nonzero_idx) > 0:
            expr_nonzero = expr[nonzero_idx]
            sorted_idx = np.argsort(-expr_nonzero)  # 降序排序
            top_genes = gene_names[nonzero_idx][sorted_idx]
            top_values = expr_nonzero[sorted_idx]
        else:
            top_genes = []
            top_values = []

        # 然后在主循环中使用：
        x1 = adata.obs.loc[cell_id, 'imagerow']
        y1 = adata.obs.loc[cell_id, 'imagecol']
        x2 = adata.obs.loc[cell_id, 'x']
        y2 = adata.obs.loc[cell_id, 'y']
        cell_info = {
            "cell_id": cell_id,
            "spatial_image":[x1,y1],
            "spatial": [x2, y2],
            "cell_genes": top_genes,
            "cell_expr" : top_values,
            "length": len(top_genes),
            "timepoint": timepoint,
            "celltype":celltype
        }
        cell_list.append(cell_info)
        # print(cell_info)
    cell_Dataset = datasets.Dataset.from_list(cell_list)
    cell_Dataset.save_to_disk('./dataSets/all_cells.datasets')
    print(cell_Dataset)
    pass

def data_map(dataSets_path = '/home/wwd/codebox/ODT-main-final/dataSets/all_cells.datasets'):
    cell_Dataset = datasets.Dataset.load_from_disk(dataSets_path)
    print(cell_Dataset)
    print(Counter(cell_Dataset['celltype']))
    print(Counter(cell_Dataset['timepoint']))
    # Counter({'Nb2': 37064, 'epidermal': 23533, 'gut': 19704, 'parenchymal': 15279, 'cathepsin_cells': 13655, 'neural': 9064, 'muscle': 7397, 'pharynx': 5446, 'protonephridia': 1812})
    # Counter({'WT': 35284, '3d': 22093, '5d': 19789, '1.5d': 18698, '10d': 14507, '0h': 11835, '12h': 10748})
    celltype_map = {'Nb2': 0, 'epidermal': 1, 'gut': 2, 'parenchymal': 3, 'cathepsin_cells': 4, 'neural': 5,
                    'muscle': 6,
                    'pharynx': 7, 'protonephridia': 8}
    timepoint_map = {'WT': 0, '3d': 1, '5d': 2, '1.5d': 3, '10d': 4, '0h': 5, '12h': 6}
    token_path = '/home/wwd/codebox/ODT-main-final/dataSets/gene_tk.pkl'
    with open(token_path, 'rb') as file:
        token_id = pickle.load(file)
    def map_id(example):
        example['input_ids'] = [token_id[gene] for gene in example['cell_genes'] if gene in token_id][:2048]
        example['label'] = timepoint_map[example['timepoint']]
        example['length'] = len(example['input_ids'])
        return example
    cell_Dataset = cell_Dataset.map(map_id, num_proc= 16)
    cell_Dataset = cell_Dataset.shuffle()
    return cell_Dataset
def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions[0] \
        if isinstance(pred.predictions, tuple) \
        else pred.predictions

    preds = logits.argmax(-1)
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
def get_snap_shot(adata_path = '/home/wwd/codebox/ODT-main-final/dataSets/spatial_unified_domain_50.h5ad'):
    # 读取 h5ad 文件
    adata = sc.read(adata_path)
    # 打印 AnnData 对象的一些信息
    print(adata)
    print(adata.obs['timepoint'])

    # 获取所有 `timepoint` 为 'WT' 的细胞子集
    adata_wt = adata[adata.obs['timepoint'] == 'WT', :]

    # 打印子集数据的信息，查看是否正确筛选
    print(adata_wt)
    # 获取空间位置数据 (x, y)
    spatial_positions = adata_wt.obs['xy']
    # 使用 .str.split() 方法将字符串拆分为两个部分，并转换为整数
    coords_split = spatial_positions.str.split(',', expand=True)

    # 将拆分出的坐标转换为浮点数
    x_coords = coords_split[0].astype(int)  # x 坐标
    y_coords = coords_split[1].astype(int)  # y 坐标

    # 获取坐标的最小值和最大值
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    # 将坐标归一化到 [0, 1] 范围内
    x_normalized = (x_coords - x_min) / (x_max - x_min)
    y_normalized = (y_coords - y_min) / (y_max - y_min)

    # 将归一化后的坐标映射到 [0, 299] 范围内
    x_mapped = (x_normalized * 299).astype(int)
    y_mapped = (y_normalized * 299).astype(int)
    # 获取所有基因的名称
    gene_names = adata_wt.var['smesg'].values

    # 创建一个字典，用于存储每个基因的空间表达快照
    gene_expression_snapshots = {}

    # 遍历每个基因，生成其在空间中的快照
    for gene in tqdm(gene_names):
        # 确保 raw 数据层存在
        if adata_wt.raw is not None:
            # 使用 raw 数据层
            gene_expression = adata_wt.raw[:, gene].X
        else:
            # 如果没有 raw 数据层，使用 adata.X 代替
            gene_expression = adata_wt[:, gene].X
        gene_expression = gene_expression.A.flatten()  # 转为一维数组

        # 创建一个 300x300 的空网格（初始值为0）
        grid = np.zeros((300, 300))

        # 填充网格，每个细胞的坐标位置填充其基因表达量
        for i in range(len(x_mapped)):
            grid[x_mapped[i], y_mapped[i]] += gene_expression[i]

        # 将该基因的空间快照存入字典
        gene_expression_snapshots[gene] = grid

    # 打印某个基因的空间快照
    print(gene_expression_snapshots[gene_names[0]])  # 打印第一个基因的快照
    # 展平每个基因的空间快照并存入列表
    flattened_snapshots = []

    # 遍历每个基因，展平其空间快照
    for gene in gene_names:
        # 获取该基因的空间快照
        grid = gene_expression_snapshots[gene]

        # 将 300x300 网格展平为一维向量
        flattened_snapshots.append(grid.flatten())
    with open('/home/wwd/codebox/ODT-main-final/dataSets/gene_expression_snapshots.pkl', 'wb') as f:
        pickle.dump(gene_expression_snapshots,f)
    return gene_names,flattened_snapshots
# 计算两个基因嵌入之间的欧几里得距离
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)
# 计算路径的总距离
def total_distance(ordered_genes, gene_dict):
    dist = 0
    for i in range(len(ordered_genes) - 1):
        dist += euclidean_distance(gene_dict[ordered_genes[i]], gene_dict[ordered_genes[i + 1]])
    # 返回环状路径（从最后一个基因回到第一个基因）
    dist += euclidean_distance(gene_dict[ordered_genes[-1]], gene_dict[ordered_genes[0]])
    return dist

# 初始化种群：随机生成一组基因顺序
def initialize_population(genes, population_size):
    population = []
    for _ in range(population_size):
        individual = genes[:]
        random.shuffle(individual)
        population.append(individual)
    return population


# 选择操作：轮盘赌选择法
def selection(population, gene_dict):
    # 计算每个个体的适应度（路径长度，越短适应度越高）
    fitness = []
    for individual in population:
        fitness.append(1 / (total_distance(individual, gene_dict) + 1e-6))  # 加上一个小常数避免除零
    # 轮盘赌选择
    total_fitness = sum(fitness)
    probs = [fit / total_fitness for fit in fitness]
    selected = random.choices(population, probs, k=2)
    return selected


# 交叉操作：部分匹配交叉（PMX）
def crossover(parent1, parent2):
    size = len(parent1)
    # 随机选择交叉区间
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))

    # 创建交叉子代
    child1 = [-1] * size
    child2 = [-1] * size

    # 拷贝交叉区间
    child1[cxpoint1:cxpoint2] = parent1[cxpoint1:cxpoint2]
    child2[cxpoint1:cxpoint2] = parent2[cxpoint1:cxpoint2]

    # 填充剩余基因
    for i in range(size):
        if parent2[i] not in child1:
            for j in range(size):
                if child1[j] == -1:
                    child1[j] = parent2[i]
                    break
        if parent1[i] not in child2:
            for j in range(size):
                if child2[j] == -1:
                    child2[j] = parent1[i]
                    break

    return child1, child2


# 变异操作：随机交换
def mutation(individual):
    size = len(individual)
    idx1, idx2 = random.sample(range(size), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


# 遗传算法主函数
def gene_order(gene_dict, population_size=50, generations=100, mutation_rate=0.1):
    genes = list(gene_dict.keys())
    population = initialize_population(genes, population_size)

    for generation in tqdm(range(generations)):
        new_population = []

        # 选择和交叉生成新种群
        while len(new_population) < population_size:
            parent1, parent2 = selection(population, gene_dict)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])

        # 变异操作
        for i in range(population_size):
            if random.random() < mutation_rate:
                new_population[i] = mutation(new_population[i])

        # 更新种群，保留最好的个体
        population = sorted(new_population, key=lambda x: total_distance(x, gene_dict))[:population_size]

    # 最终返回最好的个体（路径最短的基因顺序）
    best_individual = min(population, key=lambda x: total_distance(x, gene_dict))
    return best_individual


def tspo_ga(xy, dmat=None, pop_size=200, num_iter=1000):
    """
    Python 版的遗传算法 TSP 优化 (伪时序, 非闭环)

    参数:
        xy : ndarray, shape (N, 2) 或 (N, 3)，点坐标
        dmat : ndarray, 距离矩阵，如果为 None 则用 xy 计算
        pop_size : 种群规模
        num_iter : 迭代次数


    返回:
        dict 包含:
            opt_route : 最优路径 (ndarray)
            min_dist : 最优路径长度
            dist_history : 每次迭代的最优解
    """
    N = xy.shape[0] # 基因数
    print(N)
    # 距离矩阵
    def pairwise_distances(xy, batch_size=1000):
        n = xy.shape[0]
        dmat = np.zeros((n, n), dtype=np.float32)
        for i in tqdm(range(0, n, batch_size)):
            end_i = min(i + batch_size, n)
            for j in range(i, n, batch_size):  # 从 i 开始
                end_j = min(j + batch_size, n)
                diff = xy[i:end_i, None, :] - xy[None, j:end_j, :]
                block = np.sqrt(np.sum(diff ** 2, axis=2))
                dmat[i:end_i, j:end_j] = block
                dmat[j:end_j, i:end_i] = block.T  # 对称填充
        return dmat
    if dmat is None:
        dmat = pairwise_distances(xy,batch_size=10)
    print(dmat[0])
    # 初始化种群
    pop = np.zeros((pop_size, N), dtype=int)
    pop[0] = np.arange(N)
    for k in range(1, pop_size):
        pop[k] = np.random.permutation(N)

    global_min = np.inf
    opt_route = pop[0].copy()
    dist_history = np.zeros(num_iter)

    # 遗传算法迭代
    for iter in tqdm(range(num_iter)):
        # 计算路径长度 (伪时序，不闭环)
        total_dist = np.array([
            np.sum(dmat[route[:-1], route[1:]])
            for route in pop
        ])

        # 找到最优解
        min_dist = total_dist.min()
        idx = total_dist.argmin()
        dist_history[iter] = min_dist
        if (iter % 100 == 0):
            print(global_min)
        if min_dist < global_min:
            global_min = min_dist
            opt_route = pop[idx].copy()
        # GA操作
        new_pop = np.zeros_like(pop)
        rand_order = np.random.permutation(pop_size)

        for p in range(3, pop_size, 4):
            routes = pop[rand_order[p - 3:p + 1]]
            dists = total_dist[rand_order[p - 3:p + 1]]
            best_route = routes[dists.argmin()].copy()

            I, J = np.sort(np.random.choice(N, 2, replace=False))

            tmp_pop = np.zeros((4, N), dtype=int)

            # k=1: 最优秀的不变
            tmp_pop[0] = best_route
            # k=2: Flip
            tmp_pop[1] = best_route.copy()
            tmp_pop[1][I:J + 1] = tmp_pop[1][I:J + 1][::-1]
            # k=3: Swap
            tmp_pop[2] = best_route.copy()
            tmp_pop[2][[I, J]] = tmp_pop[2][[J, I]]
            # k=4: Slide
            tmp_pop[3] = best_route.copy()
            if J > I:
                tmp_pop[3][I:J + 1] = np.r_[tmp_pop[3][I + 1:J + 1], tmp_pop[3][I]]

            new_pop[p - 3:p + 1] = tmp_pop

        pop = new_pop


    return {"opt_route": opt_route, "min_dist": global_min,
            "dist_history": dist_history}

def edge_class(edge_weight):
    if (edge_weight < -100):
        return 0
    if (edge_weight >= -100 and edge_weight < -15):
        return 1
    if (edge_weight >= -15 and edge_weight < -3):
        return 2
    if (edge_weight >= -3 and edge_weight <= 3):
        return 3
    if (edge_weight > 3 and edge_weight < 15):
        return 4
    if (edge_weight >= 15 and edge_weight < 100):
        return 5
    if (edge_weight >= 100):
        return 6
