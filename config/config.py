import torch


class Config(object):
    def __init__(self):
        self.bert_path = '../BERT/bert-base-chinese'
        # 如果classificer是None，则默认使用bert模型
        self.models_name = ['BiGRU_Attention', 'TextCNN', 'Bert']
        self.classifier_model_name = self.models_name[2]
        # 如果fusion是None，则默认不适用融合模型 'TGFM'
        self.fusion_model_name = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_name = 'TCM_SD'

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
                'num_labels': 11
            }

        }
        self.ge_settings = {
            'window_size': 5,
            'graph_idx_path': '../mydatasets/graph/herb_graph_idx.csv',
            'graph_word_path': '../mydatasets/graph/herb_graph_word.csv',
            'graph_embedding_path': '../mydatasets/graph/herb_graph_embeddings.csv',
            'attention_path': '../mydatasets/graph/herb_attention.csv',
            'embedding_dim': 768,
            'epoch': 400,
            'num_walks': 4,
            'gamma': 0.5,
            'beta': 0.5,
            'learning_rate': 0.005
        }

        self.training_settings = {
            'batch_size': 32,
            'learning_rate': 5e-3,
            'num_epochs': 20,
            'max_seq_len': 256,
            'embedding_dim': self.ge_settings['embedding_dim'],
            # cnn
            'num_filters': 100,
            'filter_size': [3, 4, 5],
            # rnn
            'hidden_dim': 100,
            'num_layers': 2,

            'early_stopping_patience': 10,
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
