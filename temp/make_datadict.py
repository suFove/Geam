# import os
# import pickle
#
# import h5py
# from gensim.models import word2vec, KeyedVectors
#
# from trains.models import GraphEmbeddingTrainer
#
# '''
#     make data dict
# '''
# import numpy as np
# import pandas as pd
# import torch
# from transformers import AlbertTokenizerFast, BertModel, BertTokenizer
# from datasets import load_dataset, load_from_disk
# from config.config import Config
# from utils.common import tokenize_chinese_text, save_dataloader, create_dataloader, save_data_loaders, load_dataloaders, \
#     train_word_vectors, pad_or_truncate
# from utils.util4ge import word_to_idx, idx_to_tensor
# import shutil
# import os
#
# # 指定一个较小的缓存目录
# cache_dir = "D:\\download\\cache"
# # 创建缓存目录（如果不存在）
# os.makedirs(cache_dir, exist_ok=True)
#
#
# def init_components():
#     config = Config()
#     # 加载Tokenizer和Model
#     tokenizer = BertTokenizer.from_pretrained(config.bert_path)
#     embedd_layer = BertModel.from_pretrained(config.bert_path).embeddings  # 提取Embedding层
#
#     # 名称文件
#     df_graph_word = pd.read_csv(config.ge_settings['graph_word_path'])
#     # 下标文件
#     df_graph_idx = pd.read_csv(config.ge_settings['graph_idx_path'])
#     # ge文件
#     if not os.path.exists(config.ge_settings['graph_embedding_path']):
#         print(f"file {config.ge_settings['graph_embedding_path']} is not exists, create it by AttentionWalk now...")
#         ge_trainer = GraphEmbeddingTrainer(config.ge_settings, True)
#         ge_trainer.fit()
#         ge_trainer.save_model()
#     df_graph_idx2tensor = pd.read_csv(config.ge_settings['graph_embedding_path'])
#
#     #  转化映射关系
#     word_idx_dict = word_to_idx(df_graph_word, df_graph_idx)
#     idx_tensor_dict = idx_to_tensor(df_graph_idx2tensor)
#
#     return config, tokenizer, embedd_layer, word_idx_dict, idx_tensor_dict
#
#
#
#
# class EmbeddingHandler:
#     def __init__(self, config, word_idx_dict, idx_tensor_dict):
#         self.config = config
#         self.word_idx_dict = word_idx_dict
#         self.idx_tensor_dict = idx_tensor_dict
#
#     def get_tensor_from_token(self, token):
#         idx = self.word_idx_dict.get(token)
#         if idx is not None and idx in self.idx_tensor_dict:
#             return self.idx_tensor_dict[idx]
#         return None
#
#     def map_text_to_tensors(self, tokens):
#         buf_tensors = []
#         for token in tokens:
#             tensor_embedded = self.get_tensor_from_token(token)
#             if tensor_embedded is not None:
#                 buf_tensors.append(tensor_embedded)
#
#         if len(buf_tensors) > 0:
#             np_tensor = np.stack(buf_tensors, axis=0)
#             np_tensor = pad_or_truncate(np_tensor, self.config.training_settings['max_seq_len'])
#         else:
#             np_tensor = np.zeros((self.config.training_settings['max_seq_len'],
#                                   self.config.training_settings['embedding_dim']), dtype=np.float32)
#         return np_tensor
#
#
# class MakeDataDict(object):
#     def __init__(self, config, tokenizer, embedd_layer, word_idx_dict, idx_tensor_dict):
#         self.config = config
#         self.tokenizer = tokenizer
#         self.word2vec_model = None
#         self.embedd_layer = embedd_layer
#         self.word_idx_dict = word_idx_dict
#         self.idx_tensor_dict = idx_tensor_dict
#         self.embedding_handler = EmbeddingHandler(self.config, self.word_idx_dict, self.idx_tensor_dict)
#
#     def preprocess_function(self, examples):
#         return self.tokenizer(examples['text'], truncation=True, padding='max_length',
#                               max_length=self.config.training_settings['max_seq_len'])
#
#     def tokenizer_function(self, examples):
#         tokenized_texts = [tokenize_chinese_text(text) for text in examples['text']]
#         gembeddings = [self.embedding_handler.map_text_to_tensors(tokens) for tokens in tokenized_texts]
#         return {'tokenized_text': tokenized_texts, 'gembeddings': gembeddings}
#
#     def get_word_embeddings(self, texts, max_len=256):
#         """
#         为给定文本列表中的每个单词计算词嵌入，并返回句子级别的嵌入。
#
#         参数:
#             texts (list of str): 文本列表，其中每个元素是一个包含空格分隔的单词字符串。
#
#         返回:
#             embeddings (torch.Tensor): 对应于每个文本中每个单词的嵌入表示。
#         """
#         embeddings_list = []
#         for text in texts:
#             words = text
#             sentence_embeddings = []
#             for word in words:
#                 if word in self.word2vec_model.key_to_index:
#                     sentence_embeddings.append(self.word2vec_model[word])
#
#             # 如果句子中有任何有效单词，则填充或截断，否则返回一个零向量。
#             if len(sentence_embeddings) > 0:
#                 padded_embedding = pad_or_truncate(np.stack(sentence_embeddings, axis=0), max_len)
#                 embeddings_list.append(padded_embedding)
#             else:
#                 zero_vector = np.zeros((max_len, self.word2vec_model.vector_size))
#                 embeddings_list.append(zero_vector)
#
#         return np.stack(embeddings_list, axis=0)
#
#     def get_dataloader(self):
#         data_files = {
#             "train": self.config.dataset_info[self.config.dataset_name]['train_path'],
#             "dev": self.config.dataset_info[self.config.dataset_name]['dev_path'],
#             "test": self.config.dataset_info[self.config.dataset_name]['test_path']
#         }
#         dataset = load_dataset("csv", data_files=data_files)
#
#         # 预处理数据，生成 input_ids, token_type_ids 和 attention_mask
#         encoded_dataset = dataset.map(self.preprocess_function, batched=True)
#
#         # 添加 graph field 和 xembeddings
#
#         encoded_dataset['train'] = encoded_dataset['train'].map(
#             lambda examples: self.tokenizer_function(examples),
#             batched=True,
#         )
#
#         # train vec model
#         self.word2vec_model = train_word_vectors(encoded_dataset['train']['tokenized_text'],
#                                                  self.config.word2vec_settings)
#
#         for split in ['train', 'dev', 'test']:
#             encoded_dataset[split] = encoded_dataset[split].map(
#                 lambda examples: {"xembeddings": self.get_word_embeddings(examples['tokenized_text'],
#                                                                           self.config.training_settings[
#                                                                               'max_seq_len'])},
#                 batched=True
#             )
#         return encoded_dataset
#
#
# '''
#     note: the batch size may be changed
# '''
#
#
# def run_make_datadict():
#     config, tokenizer, embedd_layer, word_idx_dict, idx_tensor_dict = init_components()
#
#     dd = MakeDataDict(config, tokenizer, embedd_layer, word_idx_dict, idx_tensor_dict)
#     data_dict = dd.get_dataloader()
#     print(data_dict)
#     data_dict.save_to_disk(config.dataset_info[config.dataset_name]['root_path'])
#     print("data dict saved.")
#
#     # train_loader, dev_loader, test_loader = create_dataloader(data_dict,
#     #                                                           config.training_settings['batch_size'])
#     # save_data_loaders(train_loader, dev_loader, test_loader, config)
#     # print("data loaders saved.")
#
#
# def check4dataloaders():
#     config = Config()
#     dataset_dict = load_from_disk(config.dataset_info[config.dataset_name]['root_path'])
#     print(type(dataset_dict))
#     with open(f'{config.dataset_info[config.dataset_name]['root_path']}data_dict.pkl', 'wb') as f:
#         pickle.dump(dataset_dict, f)
#
#     dataset = dataset_dict['train']
#     torch.save(dataset_dict, 'processed_dataset.pt')
#
#     # train_loader, dev_loader, test_loader = create_dataloader(dataset_dict,
#     #                                                           config.training_settings['batch_size'])
#     #
#     # for data in train_loader:
#     #     print(data['input_ids'].shape)
#     #     print(data['token_type_ids'].shape)
#     #     print(data['gembeddings'].shape)
#     #     print(data['xembeddings'].shape)
#     #     print(data['xembeddings'])
#     #     break
#
#
# # 主程序入口
# if __name__ == '__main__':
#     # run_make_datadict()
#     check4dataloaders()
