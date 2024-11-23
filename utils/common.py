import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from config.config import Config
from trains.models import TextCNN, BiGRU, TextGraphFusionModule, BiGRU_Attention, GraphEmbeddingTrainer, ClassifierBERT
from gensim.models import word2vec, Word2Vec
from utils.util4ge import word_to_idx, idx_to_tensor


def split_dataset(input_file, train_ratio=0.8, val_ratio=0.10, test_ratio=0.10):
    # 确保比例和为1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

    # 读取原始数据
    df = pd.read_csv(input_file)

    # 计算中间分割点的比例
    intermediate_train_ratio = train_ratio / (train_ratio + val_ratio)
    intermediate_test_ratio = test_ratio

    # 第一步，将数据集分为训练集+验证集和测试集
    train_val_df, test_df = train_test_split(df, test_size=intermediate_test_ratio, stratify=df['label'],
                                             random_state=2024)

    # 第二步，将训练集+验证集进一步划分为训练集和验证集
    train_df, val_df = train_test_split(train_val_df, test_size=(val_ratio / (train_ratio + val_ratio)),
                                        stratify=train_val_df['label'], random_state=2024)

    # 获取保存文件的基础路径
    root_path = input_file.replace('data.csv', '')

    # 保存到不同的CSV文件中
    train_df.to_csv(f"{root_path}train.csv", index=False)
    val_df.to_csv(f"{root_path}dev.csv", index=False)
    test_df.to_csv(f"{root_path}test.csv", index=False)


def doc_to_vec(doc, model_wv, max_seq_len, embedding_dim):
    buf_tensor = []
    for word in doc:
        if word in model_wv:
            buf_tensor.append(model_wv[word])
    np_tensor = np.stack(buf_tensor, axis=0)
    np_tensor = pad_or_truncate(np_tensor, max_seq_len)
    return np_tensor


def pad_or_truncate(np_tensor, max_seq_len):
    length = np_tensor.shape[0]
    if length < max_seq_len:
        padded_tensor = np.zeros((max_seq_len, np_tensor.shape[1]), dtype=np.float32)
        padded_tensor[:length] = np_tensor
    else:
        padded_tensor = np_tensor[:max_seq_len].astype(np.float32)
    return padded_tensor


def train_word_vectors(texts_tokenized, word2vec_config):
    model = Word2Vec(sentences=texts_tokenized,
                     vector_size=word2vec_config['vector_size'],
                     window=word2vec_config['window_size'],
                     min_count=word2vec_config['min_count'],
                     epochs=word2vec_config['vector_epochs'],
                     workers=word2vec_config['num_workers'])
    model.wv.save_word2vec_format(word2vec_config['word2vec_path'],
                                  binary=True)
    return model.wv


def load_word_vectors(path_to_load):
    word_vectors = word2vec.KeyedVectors.load_word2vec_format(
        path_to_load,
        binary=True,
        unicode_errors='ignore')
    return word_vectors


def init_components():
    config = Config()
    # 加载Tokenizer和Model
    # 名称文件
    df_graph_word = pd.read_csv(config.ge_settings['graph_word_path'])
    # 下标文件
    df_graph_idx = pd.read_csv(config.ge_settings['graph_idx_path'])
    # ge文件
    if not os.path.exists(config.ge_settings['graph_embedding_path']):
        print(f"file {config.ge_settings['graph_embedding_path']} is not exists, create it by AttentionWalk now...")
        ge_trainer = GraphEmbeddingTrainer(config.ge_settings, True)
        ge_trainer.fit()
        ge_trainer.save_model()
    df_graph_idx2tensor = pd.read_csv(config.ge_settings['graph_embedding_path'])

    #  转化映射关系
    word_idx_dict = word_to_idx(df_graph_word, df_graph_idx)
    idx_tensor_dict = idx_to_tensor(df_graph_idx2tensor)

    return config, word_idx_dict, idx_tensor_dict


def get_base_model(config):
    bert_path = config.bert_path
    max_seq_len = config.training_settings['max_seq_len']
    embedding_dim = config.training_settings['embedding_dim']
    filter_size = config.training_settings['filter_size']
    num_filters = config.training_settings['num_filters']
    hidden_dim = config.training_settings['hidden_dim']
    num_layers = config.training_settings['num_layers']
    num_labels = config.dataset_info[config.dataset_name]['num_labels']
    base_model_name = config.classifier_model_name
    fusion_model_name = config.fusion_model_name
    base_model = None
    fusion_model = None

    if fusion_model_name is not None:
        fusion_model = TextGraphFusionModule()
    if base_model_name == 'TextCNN':
        base_model = TextCNN(embed_dim=embedding_dim, num_labels=num_labels, num_filters=num_filters,
                             filter_sizes=filter_size)

    elif base_model_name == 'BiGRU':
        base_model = BiGRU(embed_dim=embedding_dim, num_labels=num_labels, hidden_dim=hidden_dim, num_layers=num_layers)
    elif base_model_name == 'BiGRU_Attention':
        base_model = BiGRU_Attention(embedding_dim, hidden_dim, num_labels, num_layers)
    elif base_model_name == 'Bert':
        base_model = ClassifierBERT(bert_path, embedding_dim, num_labels)

    else:
        base_model = None

    return base_model, fusion_model


# def dataloader2flatten(dataloader):
#     all_features = []
#     all_labels = []
#     for batch in dataloader:
#         x = batch['x']
#         y = batch['y']
#         # 展平特征，形状从 [batch_size, seq_len, embedding_dim] 变为 [batch_size, seq_len * embedding_dim]
#         flattened_features = x.view(-1, x.shape[-1])
#         all_features.append(flattened_features.cpu().numpy())
#         all_labels.extend(y.cpu().numpy())
#     return all_features, all_labels
def dataloader2flatten(loader):
    all_x, all_y = [], []
    for batch in loader:
        x = batch['x']
        y = batch['y']
        x = x.cpu().numpy()  # 将tensor转换为numpy数组，并确保在CPU上运行
        y = y.cpu().numpy()
        all_x.extend(x.reshape(x.shape[0], -1))
        all_y.extend(y.reshape(y.shape[0], -1))
    return np.array(all_x), np.array(all_y)

def compute_metrics(y_true, y_pred):
    # 计算准确率
    acc = accuracy_score(y_true, y_pred)

    # 计算精确率、召回率和F1分数，average='weighted'表示使用加权平均
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro',
                                                               zero_division=1)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


