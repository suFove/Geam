import torch


class Config(object):
    def __init__(self):
        # -------------------此处为必须的配置-------------------
        self.bert_path = "../BERT/chinese_roberta_L-8_H-512"
        # 如果classificer是None，则默认使用bert模型
        self.models_name = ['Bert_medium', 'BiGRU_Attention', 'TextCNN']
        self.classifier_model_name = self.models_name[2]
        # 如果fusion是None，则默认不适用融合模型'TGFM'
        self.methods = ['Concat', 'Add', 'TGFM']
        self.fusion_model_name = self.methods[0]
        self.dataset_name = 'TCM_SD'
        # -------------------此处为必须的配置-------------------


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


        self.dataset_info = {
            'BBCNews': {
                'root_path': '../mydatasets/BBCNews/',
                'data_path': '../mydatasets/BBCNews/data.csv',
                'train_path': '../mydatasets/BBCNews/train.csv',
                'dev_path': '../mydatasets/BBCNews/dev.csv',
                'test_path': '../mydatasets/BBCNews/test.csv',
                'num_labels': 5
            },
            'TCM_SD': {
                'root_path': '../mydatasets/TCM_SD/',
                'data_path': '../mydatasets/TCM_SD/data.csv',
                'train_path': '../mydatasets/TCM_SD/train.csv',
                'dev_path': '../mydatasets/TCM_SD/dev.csv',
                'test_path': '../mydatasets/TCM_SD/test.csv',
                'num_labels': 148
            }

        }
        self.ge_settings = {
            'window_size': 5,
            'embedding_dim': 512,
            'epoch': 400,
            'num_walks': 4,
            'gamma': 0.5,
            'beta': 0.5,
            'learning_rate': 0.01,
            'graph_idx_path': f'../mydatasets/graph/herb_graph_idx.csv',
            'graph_word_path': f'../mydatasets/graph/herb_graph_word.csv',
            'graph_embedding_path': f'../mydatasets/graph/herb_graph_embeddings.csv',
            'attention_path': f'../mydatasets/graph/herb_attention.csv',

        }

        self.training_settings = {
            'batch_size': 32,
            'learning_rate': 3e-5,
            'num_epochs': 30,
            'max_seq_len': 256,
            'embedding_dim': self.ge_settings['embedding_dim'],
            # cnn
            'num_filters': 100,
            'filter_size': [3, 4, 5],
            # rnn
            'hidden_dim': 100,
            'num_layers': 2,

            'early_stopping_patience': 5,
            'out_dir': f'../result/{self.dataset_name}/{self.classifier_model_name}/{self.fusion_model_name}/'
        }

        self.word2vec_settings = {
            'word2vec_path': f"../word_vectors/{self.dataset_name}_{self.training_settings['embedding_dim']}.bin",
            'vector_size': self.training_settings['embedding_dim'],
            'min_freq': 5,
            'window_size': 5,
            'min_count': 5,
            'vector_epochs': 50,
            'num_workers': 4
        }
