import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, AutoModelForSequenceClassification, BertModel

from config.config import Config
from utils.berts import compute_metrics
from utils.common import create_dataloader, dataloader2flatten


def init_components():
    """Initialize"""
    config = Config()
    print(f'Current dataset is: {config.dataset_name}')
    print(f"The category is: {config.dataset_info[config.dataset_name]['num_labels']}")

    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(config.bert_path,
                                                                    num_labels=config.dataset_info[config.dataset_name][
                                                                        'num_labels'])
    embedding_layer = BertModel(bert_model.config).embeddings

    print("Loading dataset from disk")
    dataset_dict = load_from_disk(config.dataset_info[config.dataset_name]['root_path'])

    train_loader, dev_loader, test_loader = create_dataloader(dataset_dict,
                                                              tokenizer,
                                                              embedding_layer,
                                                              config.training_settings['batch_size'])
    print("Loading finished")
    print("Loading dataset from disk")
    train_features, train_labels = dataloader2flatten(train_loader)
    dev_features, dev_labels = dataloader2flatten(dev_loader)
    test_features, test_labels = dataloader2flatten(dev_loader)

    return train_features, train_labels, dev_features, dev_labels, test_features, test_labels


def run():
    train_features, train_labels, dev_features, dev_labels, test_features, test_labels = init_components()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_features)
    X_dev_scaled = scaler.transform(dev_features)

    models = {
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier(),
        'LR': LogisticRegression(),
        'NB': GaussianNB(),
    }

    # 训练和评估模型
    for name, model in models.items():
        model.fit(X_train_scaled, train_labels.ravel())
        y_dev_pred = model.predict(X_dev_scaled)

def evaluateML(y_pred, y_true):
    metrics = compute_metrics(y_pred, y_true)
    return metrics
