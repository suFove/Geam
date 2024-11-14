import os

from trains.models import GraphEmbeddingTrainer

'''
    make data dict
'''
import numpy as np
import pandas as pd
import torch
from transformers import AlbertTokenizerFast, BertModel, BertTokenizer
from datasets import load_dataset
from config.config import Config
from utils.common import tokenize_chinese_text, save_dataloader, create_dataloader
from utils.util4ge import word_to_idx, idx_to_tensor


def init_components():
    config = Config()
    # 加载Tokenizer和Model
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    embedd_layer = BertModel.from_pretrained(config.bert_path).embeddings  # 提取Embedding层

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

    return config, tokenizer, embedd_layer, word_idx_dict, idx_tensor_dict


class EmbeddingHandler:
    def __init__(self, config, word_idx_dict, idx_tensor_dict):
        self.config = config
        self.word_idx_dict = word_idx_dict
        self.idx_tensor_dict = idx_tensor_dict

    def pad_or_truncate(self, np_tensor, max_seq_len):
        length = np_tensor.shape[0]
        if length < max_seq_len:
            padded_tensor = np.zeros((max_seq_len, np_tensor.shape[1]), dtype=np.float32)
            padded_tensor[:length] = np_tensor
        else:
            padded_tensor = np_tensor[:max_seq_len].astype(np.float32)
        return padded_tensor

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
            np_tensor = self.pad_or_truncate(np_tensor, self.config.training_settings['max_seq_len'])
        else:
            np_tensor = np.zeros((self.config.training_settings['max_seq_len'],
                                  self.config.training_settings['embedding_dim']), dtype=np.float32)
        return torch.from_numpy(np_tensor)


class MakeDataDict(object):
    def __init__(self, config, tokenizer, embedd_layer, word_idx_dict, idx_tensor_dict):
        self.config, self.tokenizer, self.embedd_layer, self.word_idx_dict, self.idx_tensor_dict = config, tokenizer, embedd_layer, word_idx_dict, idx_tensor_dict
        self.embedding_handler = EmbeddingHandler(self.config, self.word_idx_dict, self.idx_tensor_dict)

    # 编码函数，格式化数据集
    def preprocess_function(self, examples):
        return self.tokenizer(examples['text'], truncation=True, padding='max_length',
                              max_length=self.config.training_settings['max_seq_len'], )

    # 编码函数，格式化数据集，添加 graph embed field
    def tokenizer_function(self, examples):
        tokenized_texts = [tokenize_chinese_text(text) for text in examples['text']]
        # 生成 gembeddings
        gembeddings = [self.embedding_handler.map_text_to_tensors(tokens) for tokens in tokenized_texts]
        return {'tokenized_text': tokenized_texts, 'gembeddings': gembeddings}

    # 获取数据集
    def get_dataloader(self):
        data_files = {"train": self.config.dataset_info[self.config.dataset_name]['train_path'],
                      "dev": self.config.dataset_info[self.config.dataset_name]['dev_path'],
                      "test": self.config.dataset_info[self.config.dataset_name]['test_path']}
        dataset = load_dataset("csv", data_files=data_files)
        # print('original data', dataset)
        # 预处理数据, 编码
        encoded_dataset = dataset.map(self.preprocess_function, batched=True)
        # print('tokenized_dataset:', encoded_dataset)
        # 添加graph field
        encoded_dataset['train'] = encoded_dataset['train'].map(
            lambda examples: self.tokenizer_function(examples),
            batched=True,
        )
        # print('encoded_dataset', encoded_dataset)
        return encoded_dataset


'''
    note: the batch size may be changed
'''
def run_make_datadict():
    config, tokenizer, embedd_layer, word_idx_dict, idx_tensor_dict = init_components()
    dd = MakeDataDict(config, tokenizer, embedd_layer, word_idx_dict, idx_tensor_dict)
    data_dict = dd.get_dataloader()
    print(data_dict)
    data_dict.save_to_disk(config.dataset_info[config.dataset_name]['root_path'])
    print("data dict saved.")


# 主程序入口
if __name__ == '__main__':
    run_make_datadict()


