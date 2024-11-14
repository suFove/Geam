import dill
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertTokenizerFast, BertModel, BertTokenizer
from datasets import load_dataset

from config.config import Config
from utils.berts import EmbeddingDataset
from utils.common import tokenize_chinese_text, save_dataloader, create_dataloader
from utils.util4ge import word_to_idx, idx_to_tensor

# 假设Config是一个包含路径、训练设置等的配置类
config = Config()

# 加载Tokenizer和Model
tokenizer = BertTokenizer.from_pretrained(config.bert_path)
model = BertModel.from_pretrained(config.bert_path)
embedd_layer = model.embeddings  # 提取Embedding层

# 名称文件
df_graph_word = pd.read_csv(config.ge_settings['graph_word_path'])
# 下标文件
df_graph_idx = pd.read_csv(config.ge_settings['graph_idx_path'])
# ge文件
df_graph_idx2tensor = pd.read_csv(config.ge_settings['graph_embedding_path'])

#  转化映射关系
word_idx_dict = word_to_idx(df_graph_word, df_graph_idx)
idx_tensor_dict = idx_to_tensor(df_graph_idx2tensor)



class EmbeddingHandler:
    def __init__(self, word_idx_dict, idx_tensor_dict):
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
            np_tensor = self.pad_or_truncate(np_tensor, config.training_settings['max_seq_len'])
        else:
            np_tensor = np.zeros((config.training_settings['max_seq_len'],
                      config.training_settings['embedding_dim']), dtype=np.float32)
        return torch.from_numpy(np_tensor)



# 编码函数，格式化数据集
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length',
                     max_length=config.training_settings['max_seq_len'], )


# 编码函数，格式化数据集
def tokenizer_function(examples, embedding_handler: EmbeddingHandler):
    tokenized_texts = [tokenize_chinese_text(text) for text in examples['text']]

    # 生成 gembeddings
    gembeddings = [embedding_handler.map_text_to_tensors(tokens) for tokens in tokenized_texts]

    return {'tokenized_text': tokenized_texts, 'gembeddings': gembeddings}


# 获取数据集
def get_dataloader():
    data_files = {"train": config.dataset_info[config.dataset_name]['train_path'],
                  "dev": config.dataset_info[config.dataset_name]['dev_path'],
                  "test": config.dataset_info[config.dataset_name]['test_path']}
    dataset = load_dataset("csv", data_files=data_files)
    print(dataset)

    # 预处理数据, 编码
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    print('encoded_ datasets:', encoded_dataset)
    print(encoded_dataset['dev']['text'])

    # 初始化 EmbeddingHandler, 对训练集进行增强
    embedding_handler = EmbeddingHandler(word_idx_dict, idx_tensor_dict)

    encoded_dataset['train'] = encoded_dataset['train'].map(
        lambda examples: tokenizer_function(examples, embedding_handler),
        batched=True,
    )

    print(encoded_dataset)
    return encoded_dataset


# 将数据集转换为DataLoader





# 主程序入口
if __name__ == '__main__':
    # 获取数据集并处理
    encoded_dataset = get_dataloader()

    print(encoded_dataset)
    # encoded_dataset.save_to_disk(config.dataset_info[config.dataset_name]['root_path'])
    # # 转换为DataLoader
    trainloader, devloader, testloader = create_dataloader(encoded_dataset, config.training_settings['batch_size'])
    save_dataloader(config.dataset_info[config.dataset_name]['root_path']+ 'train.pkl', trainloader)
    save_dataloader(config.dataset_info[config.dataset_name]['root_path'] + 'dev.pkl', devloader)
    save_dataloader(config.dataset_info[config.dataset_name]['root_path'] + 'test.pkl', testloader)


    # 打印批量数据
    # for batch in dataloader:
    #     print(batch['xembeddings'].shape)  # 打印Embedding的形状
    #     print(batch['gembeddings'].shape)
    #     print(batch['labels'].shape)  # 打印Embedding的形状
    #     # print(batch['text'])
    #     break  # 仅打印一个batch
