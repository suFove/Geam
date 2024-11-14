import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import get_linear_schedule_with_warmup

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


class CustomTrainer:
    def __init__(self, classifier_model, training_args, train_dataloader, eval_dataloader, compute_metrics,
                 feature_fusion_model=None, device='cpu'):
        self.device = torch.device(device)
        self.classifier_model = classifier_model.to(self.device)

        # Decide if using a feature fusion model
        self.fuse_feature = feature_fusion_model is not None
        self.feature_fusion_model = feature_fusion_model
        if self.fuse_feature:
            self.feature_fusion_model = self.feature_fusion_model.to(self.device)

        # Training and Evaluation Dataloaders, Metrics etc.
        self.training_args = training_args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics
        self.output_dir = training_args['out_dir']

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.classifier_model.parameters(), lr=training_args['learning_rate'])
        total_steps = len(train_dataloader) * training_args['num_epochs']
        warmup_steps = len(train_dataloader) // 4
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps)

    def train(self):
        if self.fuse_feature:
            self.feature_fusion_model.train()
        self.classifier_model.train()
        best_val_loss = float('inf')
        total_loss, avg_train_loss, val_metrics, early_stop_counter = 0, 0, 0, 0
        for epoch in range(self.training_args['num_epochs']):
            for batch in self.train_dataloader:
                x = batch['xembeddings'].to(self.device)
                g = batch['gembeddings'].to(self.device) if 'gembeddings' in batch else None
                y = batch['labels'].to(self.device)

                if self.fuse_feature and g is not None:
                    x = self.feature_fusion_model(x, g)

                self.optimizer.zero_grad()
                outputs = self.classifier_model(x)
                loss = self.loss_fn(outputs, y)
                total_loss += loss.item()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.classifier_model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

            # Evaluate the model
            val_metrics, val_loss = self.evaluate(eval_dataloader=self.eval_dataloader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            print(f"Validation Loss: {val_loss:.4f}")
            print(val_metrics)

            if early_stop_counter >= self.training_args['early_stopping_patience']:
                print("Early stopping triggered.")
                break

        return val_metrics, best_val_loss

    def evaluate(self, eval_dataloader):
        self.classifier_model.eval()
        predictions = []
        labels = []
        total_loss = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                x = batch['xembeddings'].to(self.device)
                y = batch['labels'].to(self.device)

                outputs = self.classifier_model(x)
                loss = self.loss_fn(outputs.logits, y)
                total_loss += loss.item()

                predictions.append(outputs.logits.cpu().numpy())
                labels.append(y.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        labels = np.concatenate(labels, axis=0)

        avg_val_loss = total_loss / len(eval_dataloader)
        metrics = self.compute_metrics((predictions, labels))

        return metrics, avg_val_loss

    def save_metrics(self, file_name, metrics):
        output_file_path = f'{self.output_dir}/{file_name}.json'
        with open(output_file_path, 'w') as f:
            json.dump(metrics, f)
        print(f'Metrics saved to {output_file_path}')