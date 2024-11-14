from trains.models import TextCNN


class Config(object):
    def __init__(self):
        self.bert_path = '../BERT/ZY-BERT'

        self.training_settings = {
            'batch_size': 16,
            'learning_rate': 2e-5,
            'num_epochs': 15,
            'max_seq_len': 256,
            'embedding_dim': 1024,
            # cnn
            'num_filters' : 100,
            'filter_size': [3,4,5],
            # rnn
            'hidden_dim': 100,
            'num_layers': 2,
        }
        self.classifier_model_name = 'TextCNN'



        self.dataset_name = 'TCM_SD'
        self.result_dir = '../results/SD_Bert/'

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
            'embedding_dim': 1024,
            'epoch': 400,
            'num_walks': 4,
            'gamma': 0.5,
            'beta': 0.5,
            'learning_rate': 0.01
        }
