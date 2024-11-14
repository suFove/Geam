import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm


def read_graph(graph_path):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers.
    :param args: Arguments object.
    :return graph: graph.
    """
    print("\nTarget matrix creation started.\n")
    graph = nx.from_edgelist(pd.read_csv(graph_path).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def feature_calculator(graph, window_size=5):
    """
    Calculating the feature tensor.
    :param args: Arguments object.
    :param graph: NetworkX graph.
    :return target_matrices: Target tensor.
    """
    index_1 = [edge[0] for edge in graph.edges()] + [edge[1] for edge in graph.edges()]
    index_2 = [edge[1] for edge in graph.edges()] + [edge[0] for edge in graph.edges()]
    values = [1 for edge in index_1]
    node_count = max(max(index_1) + 1, max(index_2) + 1)
    adjacency_matrix = sparse.coo_matrix((values, (index_1, index_2)),
                                         shape=(node_count, node_count),
                                         dtype=np.float32)
    degrees = adjacency_matrix.sum(axis=0)[0].tolist()
    degs = sparse.diags(degrees, [0])
    normalized_adjacency_matrix = degs.power(-1 / 2).dot(adjacency_matrix).dot(degs.power(-1 / 2))
    target_matrices = [normalized_adjacency_matrix.todense()]
    powered_A = normalized_adjacency_matrix
    if window_size > 1:
        for power in tqdm(range(window_size - 1), desc="Adjacency matrix powers"):
            powered_A = powered_A.dot(normalized_adjacency_matrix)
            to_add = powered_A.todense()
            target_matrices.append(to_add)
    target_matrices = np.array(target_matrices)
    return target_matrices


def adjacency_opposite_calculator(graph):
    """
    Creating no edge indicator matrix.
    :param graph: NetworkX object.
    :return adjacency_matrix_opposite: Indicator matrix.
    """
    adjacency_matrix = sparse.csr_matrix(nx.adjacency_matrix(graph), dtype=np.float32).todense()
    adjacency_matrix_opposite = np.ones(adjacency_matrix.shape) - adjacency_matrix
    return adjacency_matrix_opposite


def init_idx_node_mapping(graph_nx):
    '''
        init index and node mapping
        sorted nodes keeps same index for same graph every time run
    :return: idx_to_node, node_to_idx
    '''
    idx_to_node = {}
    node_to_idx = {}
    nodes_list = list(sorted(graph_nx.nodes()))
    for idx, node in enumerate(nodes_list):
        idx_to_node[idx] = node
        node_to_idx[node] = idx
    return idx_to_node, node_to_idx


def word_to_idx(df_graph_word, df_graph_idx):
    mapping_dict = {}
    # 遍历 df1 中的每一行，并根据其值在 df2 中查找对应的数值，然后将这些值存入 mapping_dict
    for idx in range(len(df_graph_word)):
        row1 = df_graph_word.iloc[idx]
        row2 = df_graph_idx.iloc[idx]
        node_1_value = row2['node_1']
        node_2_value = row2['node_2']
        # 将 df1 中的节点名称作为键，df2 中对应的数值作为值存入字典
        mapping_dict[row1['node_1']] = node_1_value
        mapping_dict[row1['node_2']] = node_2_value
    return mapping_dict


def idx_to_tensor(df_graph_idx2tensor):
    # 创建一个空字典
    result_dict = {}
    # 遍历 DataFrame 的每一行
    for idx, row in df_graph_idx2tensor.iterrows():
        # 将 id 列转换为整数类型并作为键
        key = int(row['id'])
        # 选择从 x_0 到 x_199 的列，并将其转换为 NumPy 数组
        values = np.array([row[col] for col in df_graph_idx2tensor.columns if col.startswith('x_')])
        # 将结果存储在字典中
        result_dict[key] = values
    return result_dict


def get_tensor_from_token(token, word_idx_dict, idx_tensor_dict):
    idx = word_idx_dict[token] if token in word_idx_dict else None
    if idx is None:
        return None
    else:
        # 查找 对应的 tensor
        tensor_embedded = idx_tensor_dict[idx] if idx in idx_tensor_dict else None
        if tensor_embedded is not None:
            return tensor_embedded
    return None