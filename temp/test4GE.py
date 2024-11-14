import pandas as pd

from config.config import Config
from trains.models import GraphEmbeddingTrainer
from utils.util4ge import word_to_idx, idx_to_tensor

config = Config()

ge_trainer = GraphEmbeddingTrainer(config.ge_settings, True)
ge_trainer.fit()
ge_trainer.save_model()

# 名称文件
df_graph_word = pd.read_csv(config.ge_settings['graph_word_path'])
# 下标文件
df_graph_idx = pd.read_csv(config.ge_settings['graph_idx_path'])
# ge文件
df_graph_idx2tensor = pd.read_csv(config.ge_settings['graph_embedding_path'])

#  转化映射关系
word_idx_dict = word_to_idx(df_graph_word, df_graph_idx)
idx_tensor_dict = idx_to_tensor(df_graph_idx2tensor)
print(word_idx_dict)
print(len(idx_tensor_dict))
print(idx_tensor_dict[215])


