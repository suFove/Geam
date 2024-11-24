from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

from pre.train_vector import load_vec_model
from trains.models import BertForEmbedding, ClassifierBERT
from utils.class_hub import CustomTrainer, BertDataset, EmbeddingHandler
from utils.common import init_components, compute_metrics


# tokenizer = BertTokenizer.from_pretrained('../BERT/chinese_roberta_L-2_H-12')
# model = BertModel.from_pretrained("../BERT/chinese_roberta_L-2_H-12")
# text = "用你喜欢的任何文本替换我。"
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# print(output)


def create_bert_dataset(train_df, dev_df, test_df, tokenizer, train_g=None, max_length=256, batch_size=32, device='cpu'):
    # 创建数据集对象
    train_dataset = BertDataset(train_df, tokenizer, max_length, train_g, device=device)
    dev_dataset = BertDataset(dev_df, tokenizer, max_length, device=device)
    test_dataset = BertDataset(test_df, tokenizer, max_length, device=device)

    # 创建 DataLoader 对象
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader


def init_bert_components(config, train_df, dev_df, test_df, eh:EmbeddingHandler):
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    # 3.处理数据
    # graph embedding, df to list
    train_g = train_df['tokenized_text'].apply(lambda x: eh.map_text_to_tensors(eval(x))).tolist()
    train_loader, dev_loader, test_loader = create_bert_dataset(train_df, dev_df, test_df, tokenizer,
                                                                train_g=train_g,
                                                                max_length=config.training_settings['max_seq_len'],
                                                                batch_size=config.training_settings['batch_size'],
                                                                device=config.device)

    return train_loader, dev_loader, test_loader


# def run_bert():
#     config, trainer, train_loader, dev_loader, test_loader = init_bert_components()
#     val_metrics, best_val_loss = trainer.run_epoch()
#     print(f"Best Validation Loss: {best_val_loss}")
#     print(f"Best Validation Metrics: {val_metrics}")
#     trainer.save_metrics('val_metrics', val_metrics)
#
#     test_metrics, best_test_loss = trainer.run_evaluate(test_loader)
#     print(f"Best Test Loss: {best_val_loss}")
#     print(f"Best Test Metrics: {test_metrics}")
#     trainer.save_metrics('test_metrics', test_metrics)

if __name__ == "__main__":
    run_bert()

