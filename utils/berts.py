import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset

'''
============= 编码数据集 ================
'''
class EmbeddingDataset(Dataset):
    def __init__(self, dataset, tokenizer, embedd_layer, isTrain=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.embedd_layer = embedd_layer
        self.isTrain = isTrain
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 获取数据字典
        item = self.dataset[idx]
        #
        input_ids = torch.tensor(item['input_ids'])
        token_type_ids = torch.tensor(item['token_type_ids'])
        if self.isTrain:
            gembeddings = torch.tensor(item['gembeddings'])
        label = torch.tensor(item['label']).long()
        # 获取嵌入
        with torch.no_grad():  # 禁用梯度计算
            xembeddings = self.embedd_layer(input_ids=input_ids.unsqueeze(0),
                                           token_type_ids=token_type_ids.unsqueeze(0)).squeeze(0)
        # 返回Embedding和标签（如果有的话）
        return {
            'xembeddings': xembeddings,
            'gembeddings': gembeddings,
            'labels': label
        } if self.isTrain else {
            'xembeddings': xembeddings,
            'labels': label
        }


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predicted_labels = np.argmax(predictions, axis=-1)

    # 计算准确率
    acc = accuracy_score(labels, predicted_labels)

    # 计算精确率、召回率和F1分数，average='weighted'表示使用加权平均
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted_labels, average='weighted')

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }