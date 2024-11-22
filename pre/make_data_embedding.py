import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pre.train_vector import load_vec_model
from config.config import Config
from trains.models import GraphEmbeddingTrainer
from utils.common import pad_or_truncate, doc_to_vec
from utils.util4ge import word_to_idx, idx_to_tensor


class EmbeddingHandler:
    def __init__(self, config, word_idx_dict, idx_tensor_dict):
        self.config = config
        self.word_idx_dict = word_idx_dict
        self.idx_tensor_dict = idx_tensor_dict

    def get_tensor_from_token(self, token):
        idx = self.word_idx_dict.get(token)
        if idx is not None and idx in self.idx_tensor_dict:
            return self.idx_tensor_dict[idx]
        return None

    def map_text_to_tensors(self, tokens):
        buf_tensors = []
        for token in tokens:
            tensor_embedded = self.get_tensor_from_token(token)
            if tensor_embedded is not None:
                buf_tensors.append(tensor_embedded)

        if len(buf_tensors) > 0:
            np_tensor = np.stack(buf_tensors, axis=0)
            np_tensor = pad_or_truncate(np_tensor, self.config.training_settings['max_seq_len'])
        else:
            np_tensor = np.zeros((self.config.training_settings['max_seq_len'],
                                  self.config.training_settings['embedding_dim']), dtype=np.float32)
        return np_tensor


class TextDataset(Dataset):
    def __init__(self, x, y=None, g=None):
        self.x = x
        self.y = y
        if g is not None:
            self.g = g
        else:
            self.g = [0] * len(x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        sample = {
            'x': torch.tensor(self.x[idx], dtype=torch.float32),
            'y': torch.tensor(self.y[idx], dtype=torch.long),
            'g': torch.tensor(self.g[idx], dtype=torch.float32)
        }
        return sample


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


def word_to_feature(config, word_idx_dict, idx_tensor_dict):
    all_data, train_data, dev_data, test_data, word2vec_model = load_vec_model(config)
    eh = EmbeddingHandler(config, word_idx_dict, idx_tensor_dict)
    # word embedding
    train_x = train_data['tokenized_text'].apply(lambda x: doc_to_vec(eval(x), model_wv=word2vec_model,
                                                                      max_seq_len=config.training_settings[
                                                                          'max_seq_len'],
                                                                      embedding_dim=config.training_settings[
                                                                          'embedding_dim']))
    dev_x = dev_data['tokenized_text'].apply(lambda x: doc_to_vec(eval(x), model_wv=word2vec_model,
                                                                  max_seq_len=config.training_settings['max_seq_len'],
                                                                  embedding_dim=config.training_settings[
                                                                      'embedding_dim']))
    test_x = test_data['tokenized_text'].apply(lambda x: doc_to_vec(eval(x), model_wv=word2vec_model,
                                                                    max_seq_len=config.training_settings['max_seq_len'],
                                                                    embedding_dim=config.training_settings[
                                                                        'embedding_dim']))
    # graph embedding
    train_g = train_data['tokenized_text'].apply(lambda x: eh.map_text_to_tensors(eval(x)))

    return (train_x.tolist(), train_data['label'].tolist(),
            dev_x.tolist(), dev_data['label'].tolist(),
            test_x.tolist(), test_data['label'].tolist(),
            train_g)


def create_dataloaders(train_x, train_y, dev_x, dev_y, test_x, test_y, train_g=None, batch_size=32):
    # 创建数据集对象
    train_dataset = TextDataset(train_x, train_y, train_g)
    dev_dataset = TextDataset(dev_x, dev_y)
    test_dataset = TextDataset(test_x, test_y)

    # 创建 DataLoader 对象
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader


if __name__ == '__main__':
    config, word_idx_dict, idx_tensor_dict = init_components()
    train_x, train_y, dev_x, dev_y, test_x, test_y, train_g = word_to_feature(config, word_idx_dict, idx_tensor_dict)

    print(len(train_x))
    print(len(train_y))

    train_loader, dev_loader, test_loader = create_dataloaders(train_x, train_y, dev_x, dev_y, test_x, test_y, train_g,
                                                               batch_size=config.training_settings['batch_size'])

    for batch in train_loader:
        print(batch['x'].shape)
        print(batch['y'].shape)
        print(batch['g'].shape)
    #
    # for batch in dev_loader:
    #     print(batch['x'])
    #     print(batch['y'])
    #     print(batch['g'])
    #     break
    #
    # print(type(train_x[0]))
    # print(type(train_g[0]))
    #
    # print(len(train_x[0]))
    # print(len(train_g[0]))
