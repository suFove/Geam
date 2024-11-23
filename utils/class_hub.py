import json
import os

import numpy as np
import torch
from torch.optim import AdamW, Adam
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from utils.common import pad_or_truncate

'''
============= 编码数据集 ================
'''

'''
    使用word2Vec的数据集
'''
class DeeplDataset(Dataset):
    def __init__(self, x, y=None, g=None, device=torch.device('cpu')):
        self.x = x
        self.y = y
        self.device = device
        if g is not None:
            self.g = g
        else:
            self.g = [0] * len(x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        sample = {
            'x': torch.tensor(self.x[idx], dtype=torch.float32, device=self.device),
            'y': torch.tensor(self.y[idx], dtype=torch.long, device=self.device),
            'g': torch.tensor(self.g[idx], dtype=torch.float32, device=self.device)
        }
        return sample


'''
    使用bert模型分类所构建的数据集
'''


class BertDataset(Dataset):
    def __init__(self, df_data, tokenizer, max_length=256, extra_features=None, device=torch.device('cpu')):
        self.device = device
        self.texts = df_data['text'].tolist()
        self.labels = torch.tensor(np.array(df_data['label'].tolist()), dtype=torch.long, device=self.device)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if extra_features is not None:
            self.extra_features = torch.tensor(np.array(extra_features), dtype=torch.long, device=self.device)
        else:
            self.extra_features = torch.zeros(len(self.texts), dtype=torch.long, device=self.device)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        extra_feature = self.extra_features[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_tensors="pt")
        return {
            'x': {
                'input_ids': encoding['input_ids'].squeeze(0).to(self.device),  # Remove the batch dimension
                'token_type_ids': encoding['token_type_ids'].squeeze(0).to(self.device),
                'attention_mask': encoding['attention_mask'].squeeze(0).to(self.device)
            },
            'g': extra_feature.to(self.device),
            'y': label
        }


'''
    用于匹配图嵌入字典， word to feature
'''


class EmbeddingHandler:
    def __init__(self, config, word_idx_dict, idx_tensor_dict):
        self.config = config
        self.word_idx_dict = word_idx_dict
        self.idx_tensor_dict = idx_tensor_dict

    def get_tensor_from_token(self, token):
        idx = self.word_idx_dict.get(token)
        if idx is not None and idx in self.idx_tensor_dict:
            return self.idx_tensor_dict[idx]
        return None

    def map_text_to_tensors(self, tokens):
        buf_tensors = []
        for token in tokens:
            tensor_embedded = self.get_tensor_from_token(token)
            if tensor_embedded is not None:
                buf_tensors.append(tensor_embedded)

        if len(buf_tensors) > 0:
            np_tensor = np.stack(buf_tensors, axis=0)
            np_tensor = pad_or_truncate(np_tensor, self.config.training_settings['max_seq_len'])
        else:
            np_tensor = np.zeros((self.config.training_settings['max_seq_len'],
                                  self.config.training_settings['embedding_dim']), dtype=np.float32)
        return np_tensor

'''
    自定义 trainner
'''
class CustomTrainer:
    def __init__(self, classifier_model, training_args, train_dataloader, eval_dataloader,
                 compute_metrics, feature_fusion_model=None, device='cpu'):

        self.device = torch.device(device)
        self.classifier_model = classifier_model.to(self.device)
        # self.embedd_layer = embedd_layer.to(self.device)
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
        self.optimizer = Adam(self.classifier_model.parameters(), lr=training_args['learning_rate'])
        total_steps = len(train_dataloader) * training_args['num_epochs']
        warmup_steps = len(train_dataloader) // 4
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)

    def train_model(self, batch_dict, train_bert=True):
        '''
            select diff batch dict by different dataloader
            if train_bert:
                batch dict should be:
        '''

    def run_epoch(self):
        best_val_loss = float('inf')
        val_metrics = None
        early_stop_counter = 0
        for epoch in range(self.training_args['num_epochs']):
            # 1.train
            self.run_train(epoch)

            # 2.Evaluate the model
            val_metrics, val_loss = self.run_evaluate(eval_dataloader=self.eval_dataloader)

            # 3.检查早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # print(f"Validation Loss: {val_loss:.4f}")
            # print(val_metrics)
            if early_stop_counter >= self.training_args['early_stopping_patience']:
                print("\nEarly stopping triggered.")
                break

        return val_metrics, best_val_loss

    def run_train(self, epoch):
        # 1. 设置train mode
        if self.fuse_feature:
            self.feature_fusion_model.train()
        self.classifier_model.train()
        # 2.准备指标
        total_loss, avg_train_loss = 0, 0
        # 使用tqdm包装train_dataloader，以便显示进度条
        train_dataloader_with_tqdm = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch}", leave=True)
        # 3.batch loop
        for batch in train_dataloader_with_tqdm:
            x = batch['x']
            g = batch['g']
            y = batch['y']
            self.optimizer.zero_grad()
            if self.fuse_feature and g is not None:
                x = self.feature_fusion_model(x, g)

            outputs = self.classifier_model(x)
            loss = self.loss_fn(outputs, y)
            torch.nn.utils.clip_grad_norm_(self.classifier_model.parameters(), 1.0)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()

            # 更新进度条，显示当前损失
            train_dataloader_with_tqdm.set_postfix(
                {'loss': f'{total_loss / (train_dataloader_with_tqdm.n + 1):.4f}'})

    def run_evaluate(self, eval_dataloader):
        self.classifier_model.eval()

        all_labels = []
        all_preds = []
        total_loss = 0
        with torch.no_grad():
            eval_with_tqdm = tqdm(eval_dataloader, desc="Evaluating", leave=True)

            for batch in eval_with_tqdm:
                x = batch['x']
                y = batch['y']

                outputs = self.classifier_model(x)
                loss = self.loss_fn(outputs, y)
                total_loss += loss.item()

                all_labels.extend(y.cpu().numpy())
                all_preds.extend(outputs.argmax(1).cpu().numpy())

                eval_with_tqdm.set_postfix(
                    {'loss': f'{total_loss / (eval_with_tqdm.n + 1):.4f}'})
        metrics = self.compute_metrics(all_labels, all_preds)

        return metrics, total_loss / len(eval_dataloader)

    def save_metrics(self, file_name, metrics):
        output_file_path = f'{self.output_dir}/{file_name}.json'
        # Ensure the directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        with open(output_file_path, 'w') as f:
            json.dump(metrics, f)
        print(f'Metrics saved to {output_file_path}')
