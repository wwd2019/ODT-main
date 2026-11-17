import scanpy as sc
import sys
import networkx as nx
import seaborn as sns
import pandas as pd
from Bio import SeqIO, Entrez
import matplotlib.pyplot as plt
CODE_PATH = "/home/wwd/codebox/ODT-main-final/"
if CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)
def Data_show(data_path = '/home/wwd/codebox/ODT-main-final/dataSets/spatial_unified_domain_50.h5ad'):
    # 展示2d数据集的数据分布
    # 读取数据
    adata = sc.read(data_path)
    # print(adata)
    # 提取坐标和元信息
    xy = adata.obs[['x', 'y']].copy()
    celltypes = adata.obs['celltype']
    timepoints = adata.obs['timepoint']

    # 获取细胞类型和颜色映射
    unique_types = celltypes.unique()
    palette = sns.color_palette("tab10", n_colors=len(unique_types))
    color_map = dict(zip(unique_types, palette))

    # 按 timepoint 分组绘图
    for tp in sorted(timepoints.unique()):
        subset_idx = timepoints == tp
        xy_sub = xy[subset_idx]
        celltypes_sub = celltypes[subset_idx]

        plt.figure(figsize=(10, 8))
        for ct in unique_types:
            mask = (celltypes_sub == ct)
            if (tp == 'WT'):
                plt.scatter(
                    xy_sub.loc[mask, 'x'],
                    xy_sub.loc[mask, 'y'],
                    s=0.5,
                    color=color_map[ct],
                    label=ct,
                    alpha=0.8
                )
            else:
                plt.scatter(
                    xy_sub.loc[mask, 'x'],
                    xy_sub.loc[mask, 'y'],
                    s=2,
                    color=color_map[ct],
                    label=ct,
                    alpha=0.8
                )

        plt.axis("equal")
        # 去掉x和y轴
        plt.axis('off')
        # 去掉title
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20, frameon=False)
        plt.tight_layout()
        plt.savefig(f'/home/wwd/codebox/ODT-main-final/results/planarian_2d_pngs/{tp}.png', bbox_inches='tight', dpi=450)  # 按tp命名保存
        plt.show()
# 使用 Entrez 查询物种的亲缘关系
def get_taxonomy(species_name):
    # Entrez.email = "XXX" # 替换为您的邮箱
    handle = Entrez.esearch(db="taxonomy", term=species_name)
    record = Entrez.read(handle)
    handle.close()

    # 获取物种的Taxonomy ID
    taxon_id = record["IdList"][0]

    # 使用 Taxonomy ID 获取详细的分类信息
    handle = Entrez.esummary(db="taxonomy", id=taxon_id)
    summary = Entrez.read(handle)
    handle.close()

    # 如果没有 Lineage 字段，则返回 Division（物种类别）或 Genus 和 Species
    if "Lineage" in summary[0]:
        taxonomy = summary[0]["Lineage"]
    else:
        # 使用 Division 或 Genus 和 Species 来构建树
        division = summary[0].get("Division", "Unknown")
        genus = summary[0].get("Genus", "Unknown")
        species = summary[0].get("Species", "Unknown")
        taxonomy = f"{genus} {species} ({division})"

    return taxonomy
# 基于物种的亲缘关系绘制树状图
def build_phylogenetic_tree(species_dict, species_counts):
    # 创建一个无向图
    G = nx.Graph()

    # 添加根节点
    G.add_node("Root")

    # 遍历字典，构建树结构
    for group, species_list in species_dict.items():
        # 添加分类群节点
        G.add_node(group)

        # 连接分类群到根节点
        G.add_edge("Root", group)

        # 添加每个物种节点并连接到对应的分类群
        for species in species_list:
            G.add_node(species)
            G.add_edge(species, group)  # 物种与分类群之间的连接

    # 获取每个物种的赋分，并将其映射为节点大小
    node_sizes = []
    for node in G.nodes:
        if node != "Root" and node in species_counts:  # 排除根节点
            # 根据赋分设置节点大小，这里将赋分做一个缩放
            node_sizes.append(species_counts[node] * 10)  # 乘以10来放大效果
        else:
            node_sizes.append(1000)  # 根节点的大小固定

    # 为了可视化，使分类群节点和物种节点有合理的分布
    pos = nx.spring_layout(G, seed=42)  # 使用spring布局，增加随机性
    plt.figure(figsize=(12, 8))

    # 绘制树状图
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="lightblue", font_size=10, font_weight="bold",
            edge_color="gray")

    # 显示图形
    plt.title("Phylogenetic Tree Based on Taxonomy and Species Scores")
    plt.show()

def build_phylogenetic_tree2(species_dict, species_counts):
    # 创建一个无向图
    G = nx.Graph()

    # 添加根节点
    root = "root"
    G.add_node(root)
    father_has = []
    # 定义物种类别及其父节点的层级关系
    # 创建结点
    for group, species_list in species_dict.items():
        # 如果该group已经在图中，添加到其上级（父级）节点
        G.add_node(group)  # 添加单一群体节点
        if '&' in group:  # 如果group中含有空格，则它是一个组合群，和其子群相连
            parent_group = group.split('&')  # 提取父群
            G.add_node(parent_group[0])
            G.add_node(parent_group[1])
    # 连接结点
    for group, species_list in species_dict.items():
        if '&' in group:  # 如果group中含有空格，则它是一个组合群，和其子群相连
            parent_group = group.split('&')  # 提取父群
            for parent in parent_group:
                G.add_edge(group, parent )  # 连接父群和组合群
                G.add_edge(parent , root)  # 如果没有父节点，连接到根节点
        else:
            G.add_node(group)  # 添加单一群体节点
            G.add_edge(group, root)  # 如果没有父节点，连接到根节点
        # 遍历该分类群下的物种
        for species in species_list:
            G.add_node(species)  # 为物种添加节点
            G.add_edge(species, group)  # 物种连接到分类群
            # 如果物种在字典中，设置节点的大小（基于物种的赋分）
            if species in species_counts:
                size = species_counts[species]
            else:
                size = 100  # 如果没有赋分，则给定默认大小
            G.nodes[species]['size'] = size  # 将物种的大小赋值

    # 为了可视化，使分类群节点在树状图中更高
    pos = nx.spring_layout(G, seed=42)  # 使用spring布局，增加随机性
    plt.figure(figsize=(12, 8))

    # 绘制树状图
    node_sizes = [G.nodes[node].get('size', 100) for node in G.nodes()]  # 获取节点大小
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="lightblue", font_size=10, font_weight="bold",
            edge_color="gray")

    # 显示图形
    plt.title("Phylogenetic Tree Based on Taxonomy")
    plt.show()


def generate_group_colors(groups):
    # 使用matplotlib的调色板（如tab10, Set1等）
    cmap = plt.get_cmap("Set1")  # 你可以选择其他调色板
    group_colors = {}

    # 根据分类群的数量，从调色板获取颜色
    for i, group in enumerate(groups):
        group_colors[group] = tuple(list(cmap(i % cmap.N))[:3])  # 循环使用调色板中的颜色

    return group_colors


# 递归计算群体和物种节点的大小
def calculate_size(G,node):
    # 如果节点是物种，返回其自身的大小
    if G.nodes[node].get('type') == 'species':
        return G.nodes[node].get('size', 1)

    # 如果节点是群体，计算群体的大小
    total_size = 0
    for neighbor in G.predecessors(node):
        total_size += calculate_size(G, neighbor)  # 递归计算邻居（子节点）的大小

    # 设置群体节点的 size
    G.nodes[node]['size'] = total_size
    return total_size
def build_phylogenetic_tree3(species_dict, species_counts):
    # 创建一个有向图
    G = nx.DiGraph()  # 有向图

    # 添加根节点
    root = "root"
    G.add_node(root)
    groups = []
    # 创建结点
    for group, species_list in species_dict.items():
        # 如果该group已经在图中，添加到其上级（父级）节点
        G.add_node(group)  # 添加单一群体节点
        if '&' in group:  # 如果group中含有空格，则它是一个组合群，和其子群相连
            parent_group = group.split('&')  # 提取父群
            for parent in parent_group:
                parent = parent.strip(' ')
                G.add_node(parent)  # 确保父群节点存在
                G.add_edge(group, parent)  # 连接组合群和父群
                G.add_edge(parent, root)  # 如果没有父节点，连接到根节点
                groups.append(parent)
                groups.append(group)

        else:
            G.add_edge(group, root)  # 没有父节点时连接根节点
            groups.append(group)

        # 遍历该分类群下的物种
        for species in species_list:
            G.add_node(species)  # 为物种添加节点
            G.add_edge(species, group)  # 物种连接到分类群

            # 如果物种在字典中，设置节点的大小（基于物种的赋分）
            if species in species_counts:
                size = species_counts[species]
            else:
                size = 1  # 如果没有赋分，则给定默认大小
            G.nodes[species]['size'] = size  # 将物种的大小赋值


    # 设置布局
    #pos = nx.spring_layout(G, k=0.5, iterations=50)  # 增加k值来增加节点间距
    pos = nx.kamada_kawai_layout(G)
    plt.figure(figsize=(12, 8))

    # 设置节点大小（根据物种赋分）
    node_sizes = [G.nodes[node].get('size', 1) * 10 for node in G.nodes()]  # 根据赋分调整节点大小
    # 物种群体的颜色映射（每个群体的物种颜色不同）
    groups = list(set(groups))
    groups.append("root")

    # 初始化节点的 'type' 和 'size' 属性
    for node in G.nodes():
        if node not in groups:
            G.nodes[node]['type'] = 'species'  # 物种节点的'type'为species
        else:
            G.nodes[node]['type'] = 'group'  # 群体节点的'type'为group
    # 计算group 的size
    # 对所有节点进行递归计算
    for group in groups:
        calculate_size(G,group)

    group_colors = generate_group_colors(groups)
    print(group_colors)
    # 保存节点数据

    # 提取节点属性
    nodes_data = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    # 将每个节点的属性分成不同的列
    # 如果某些节点缺少某个属性，NaN 将自动填充
    nodes_data.reset_index(inplace=True)
    nodes_data.rename(columns={'index': 'node_id'}, inplace=True)
    nodes_data.to_csv("/home/wwd/codebox/ODT-main-final/results/Blast_result/gene_data2//nodes.csv", index=False)

    # 保存边数据
    edges_df = pd.DataFrame(G.edges(data=True))
    edges_df.to_csv("/home/wwd/codebox/ODT-main-final/results/Blast_result/gene_data2/edges.csv", index=False)
    # 读取 CSV 文件
    nodes_data = pd.read_csv("/home/wwd/codebox/ODT-main-final/results/Blast_result/gene_data2/nodes.csv")

    # 检查列名是否包含空格
    nodes_data.columns = nodes_data.columns.str.strip()

    # 保存清理后的文件
    nodes_data.to_csv("/home/wwd/codebox/ODT-main-final/results/Blast_result/gene_data2/nodes_cleaned.csv", index=False)
def makeBox_species_ident():
    # 探究模型输出与匹配结果的关系
    species_groups_dict = get_species_groups_dict()
    workbook = load_workbook('/home/wwd/codebox/ODT-main-final/results/Blast_result/gene_data2/align_result.xlsx')  # 'your_file.xlsx' 替换为你的文件路径
    # 获取所有工作表名称
    sheet_names = workbook.sheetnames
    print(sheet_names)
    group_per_idents = {}
    # 选择一个工作表（例如，选择第一个工作表）
    sheet = workbook[sheet_names[0]]  # 或者 workbook['Sheet1']
    smed_align_result = {}
    group_per_ident = {}
    # 获取工作表中的所有数据
    for row in sheet.iter_rows(values_only=True):
        row_list = [item for item in row]
        if (row_list[0] not in smed_align_result):
            smed_align_result[row_list[0]] = []
        if(row_list[16]=='Scientific Name' or row_list[16] not in species_groups_dict): #表头数据和低质量数据跳过
            continue
        smed_align_result[row_list[0]].append(
                {'Accession_id': row_list[1], 'Per.Ident': float(row_list[2]), 'species': row_list[16],
                 'group': species_groups_dict[row_list[16]]})
        if(species_groups_dict[row_list[16]] not in group_per_idents):
            group_per_idents[species_groups_dict[row_list[16]]]=[]
        group_per_idents[species_groups_dict[row_list[16]]].append(row_list[2])
    print(group_per_idents.keys())

    # 获取 group_per_idents 的键作为 x 轴的顺序
    group_order = list(group_per_idents.keys())
    import matplotlib as mpl
    # 创建数据框
    labels = []
    values = []

    for group, value_list in group_per_idents.items():
        value_list = [item for item in value_list if float(item)>40]
        labels.extend([group] * len(value_list))  # 每个组名重复次数与列表长度相同
        values.extend(value_list)

    df = pd.DataFrame({
        'Group': labels,
        'Value': values
    })

    # 确保 'Value' 列是数值类型
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # 确保 'Group' 列是字符串类型
    df['Group'] = df['Group'].astype(str)
    # 创建一个颜色映射
    cmap = plt.get_cmap("Set3")  # 选择一个合适的颜色映射
    palette = {group: cmap(i / len(group_order)) for i, group in enumerate(group_order)}
    plt.figure(figsize=(16, 12))  # 增大图表的尺寸
    # 绘制箱型图，使用 group_order 来定义 x 轴的顺序，并使用 palette 来设定颜色
    boxplot = sns.boxplot(x='Group', y='Value', data=df, order=group_order, palette=palette)
    plt.xticks([])  # 这样做会完全去掉 X 轴的所有标签和刻度

    # 设置 x 轴标签垂直显示，隐藏所有的坐标和刻度
    plt.tick_params(axis='x', which='both',  bottom=False, top=False, labelbottom=False)  # 隐藏 x 轴的坐标和刻度

    # 创建图例，确保颜色与箱型图的颜色对应
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=24) for color in palette.values()]
    # 设置图例，增大颜色条的大小
    plt.legend(handles=handles, labels=group_order, bbox_to_anchor=(1.02, 1), loc='upper left',
               fontsize=28, handlelength=3, handleheight=3)  # 增大颜色条大小
    # 自定义 x 轴和 y 轴标题
    plt.xlabel("Taxonomy Groups", labelpad=15, fontsize=30)
    plt.ylabel("Per.Ident Value", fontsize=30)

    # 添加标题，并增加线条粗细
    plt.title("Taxonomy-Based Protein Per.Ident Distribution", fontsize=30)

    # 增加线条粗细
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    # 放大 y 轴的下标
    plt.tick_params(axis='y', labelsize=24)  # 设置 y 轴刻度字体大小
    # 显示图表
    plt.tight_layout()
    plt.savefig('/home/wwd/codebox/ODT-main-final/results/Blast_result/Taxonomy-Based-Protein-Per-Ident-Distribution.png', dpi = 300)
    plt.show()
