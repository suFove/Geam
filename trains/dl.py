from torch.utils.data import DataLoader
from utils.class_hub import DeeplDataset, EmbeddingHandler
from utils.common import doc_to_vec

'''
    the data dict should be created by ../pre/make_dataset.py
    before training the model
'''


def word_to_feature(config, train_data, dev_data, test_data, word2vec_model, eh: EmbeddingHandler):
    '''
        将word通过df.apply方法，将word在Word2Vec字典中的向量映射出来，并完成填充或者裁剪
    '''

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


def create_dl_dataloaders(train_x, train_y, dev_x, dev_y, test_x, test_y, train_g=None, batch_size=32, device='cpu'):
    '''
        创建dataloader, 使用DeeplDataset类
    '''
    # 创建数据集对象
    train_dataset = DeeplDataset(train_x, train_y, train_g, device=device)
    dev_dataset = DeeplDataset(dev_x, dev_y, device=device)
    test_dataset = DeeplDataset(test_x, test_y, device=device)

    # 创建 DataLoader 对象
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader


def init_dl_runner(config, train_df, dev_df, test_df, word2vec_model, eh:EmbeddingHandler):

    # 3.创建数据集，转为dataloader
    train_x, train_y, dev_x, dev_y, test_x, test_y, train_g = word_to_feature(config, train_df, dev_df, test_df, word2vec_model, eh)
    train_loader, dev_loader, test_loader = create_dl_dataloaders(train_x, train_y, dev_x, dev_y, test_x, test_y,
                                                                  train_g,
                                                                  batch_size=config.training_settings['batch_size'],
                                                                  device=config.device)

    print("Loading finished")

    return train_loader, dev_loader, test_loader
