import csv
import json
import dill
import jieba
import pandas as pd
from torch.utils.data import DataLoader
from trains.models import TextCNN, BiGRU, TextGraphFusionModule
from utils.berts import EmbeddingDataset
from gensim.models import word2vec, Word2Vec


def read_json_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def read_csv_to_lists(fname):
    texts = []
    labels = []
    with open(fname, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            labels.append(int(row['label']))
            texts.append(row['text'])
    return texts, labels


def tokenize_chinese_text(text):
    # 使用jieba进行中文分词
    tokens = jieba.lcut(text)
    return tokens


def split_dataset(input_file, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # 读取原始数据
    df = pd.read_csv(input_file)
    # 确保比例和为1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    # 随机打乱数据集
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    # 计算每个子集的大小
    n = len(df_shuffled)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    # 切分数据集
    train_df = df_shuffled.iloc[:train_size]
    val_df = df_shuffled.iloc[train_size:train_size + val_size]
    test_df = df_shuffled.iloc[train_size + val_size:]
    root_path = input_file.replace('data.csv', '')
    # 保存到不同的CSV文件中
    train_df.to_csv(f"{root_path}train.csv", index=False)
    val_df.to_csv(f"{root_path}dev.csv", index=False)
    test_df.to_csv(f"{root_path}test.csv", index=False)


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


def save_dataloader(file_path, encoded_dataloader):
    with open(file_path, 'wb') as f:
        dill.dump(encoded_dataloader, f)


def load_dataloader(file_path):
    with open(file_path, 'rb') as f:
        dataloader_saved = dill.load(f)
    return dataloader_saved


def create_dataloader(encoded_dataset, batch_size=16):
    # 创建自定义Dataset
    train_embedding_dataset = EmbeddingDataset(encoded_dataset['train'], isTrain=True)
    dev_embedding_dataset = EmbeddingDataset(encoded_dataset['dev'])
    test_embedding_dataset = EmbeddingDataset(encoded_dataset['test'])

    # 使用DataLoader加载数据
    train_loader = DataLoader(train_embedding_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_embedding_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_embedding_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, dev_loader, test_loader


def load_dataloaders(config):
    with open(config.dataset_info[config.dataset_name]['root_path'] + 'train.pkl', 'rb') as f:
        train_loader = dill.load(f)
    with open(config.dataset_info[config.dataset_name]['root_path'] + 'dev.pkl', 'rb') as f:
        dev_loader = dill.load(f)
    with open(config.dataset_info[config.dataset_name]['root_path'] + 'test.pkl', 'rb') as f:
        test_loader = dill.load(f)
    return train_loader, dev_loader, test_loader


def save_data_loaders(train_loader, dev_loader, test_loader, config):
    with open(config.dataset_info[config.dataset_name]['root_path'] + 'train.pkl', 'wb') as f:
        dill.dump(train_loader, f)
    with open(config.dataset_info[config.dataset_name]['root_path'] + 'dev.pkl', 'wb') as f:
        dill.dump(dev_loader, f)
    with open(config.dataset_info[config.dataset_name]['root_path'] + 'test.pkl', 'wb') as f:
        dill.dump(test_loader, f)


def get_base_model(config):
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

    else:
        base_model = None

    return base_model, fusion_model


def dataloader2flatten(dataloader):
    all_features = []
    all_labels = []
    for features, labels in dataloader:
        # 展平特征，形状从 [batch_size, seq_len, embedding_dim] 变为 [batch_size, seq_len * embedding_dim]
        flattened_features = features.view(features.size(0), -1)
        all_features.append(flattened_features.numpy())
        all_labels.extend(labels.numpy())
    return all_features, all_labels

# split_dataset('../mydatasets/BBCNews/data.csv', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
