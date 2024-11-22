import jieba
import pandas as pd
from config.config import Config
from utils.common import train_word_vectors, load_word_vectors
import json

def init_data(config):

    all_data = pd.read_csv(config.dataset_info[config.dataset_name]['data_path'])
    train_data = pd.read_csv(config.dataset_info[config.dataset_name]['train_path'])
    dev_data = pd.read_csv(config.dataset_info[config.dataset_name]['dev_path'])
    test_data = pd.read_csv(config.dataset_info[config.dataset_name]['test_path'])

    return all_data, train_data, dev_data, test_data


def add_tokenized_text(df):
    df['tokenized_text'] = df['text'].apply(lambda x: list(jieba.cut(x)))
    return df



def prepare_data(all_data, train_data, dev_data, test_data, config):
    all_data = add_tokenized_text(all_data)
    train_data = add_tokenized_text(train_data)
    dev_data = add_tokenized_text(dev_data)
    test_data = add_tokenized_text(test_data)
    train_data.to_csv(config.dataset_info[config.dataset_name]['data_path'], index=False)
    train_data.to_csv(config.dataset_info[config.dataset_name]['train_path'], index=False)
    dev_data.to_csv(config.dataset_info[config.dataset_name]['dev_path'], index=False)
    test_data.to_csv(config.dataset_info[config.dataset_name]['test_path'], index=False)

    return all_data, train_data, dev_data, test_data

def train_vec_model(config):
    all_data, train_data, dev_data, test_data = init_data(config)
    all_data, train_data, dev_data, test_dat = prepare_data(all_data, train_data, dev_data, test_data, config)
    print(train_data.shape, dev_data.shape, test_data.shape)
    all_tokenized_text = all_data['tokenized_text'].tolist()
    print(type(all_tokenized_text))
    print(all_tokenized_text[0][0])
    word2vec_model = train_word_vectors(all_tokenized_text, config.word2vec_settings)
    # word2vec_model = None
    return word2vec_model


def load_vec_model(config):
    all_data, train_data, dev_data, test_data = init_data(config)
    word2vec_model = load_word_vectors(config.word2vec_settings['word2vec_path'])
    return all_data, train_data, dev_data, test_data, word2vec_model

if __name__ == '__main__':
    config = Config()
    train_vec_model(config)
    # load_vec_modelc(config)
