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


# class BertDataset(Dataset):
#     def __init__(self, df_data, tokenizer, max_length=256, extra_features=None, device=torch.device('cpu')):
#         self.device = device
#         self.texts = df_data['text'].tolist()
#         self.labels = torch.tensor(np.array(df_data['label'].tolist()), dtype=torch.long, device=self.device)
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         if extra_features is not None:
#             self.extra_features = torch.tensor(np.array(extra_features), dtype=torch.long, device=self.device)
#         else:
#             self.extra_features = torch.zeros(len(self.texts), dtype=torch.long, device=self.device)
#
#     def __len__(self):
#         return len(self.texts)
#
#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         label = self.labels[idx]
#         extra_feature = self.extra_features[idx]
#         encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
#                                   return_tensors="pt")
#         return {
#             'x': {
#                 'input_ids': encoding['input_ids'].squeeze(0).to(self.device),  # Remove the batch dimension
#                 'token_type_ids': encoding['token_type_ids'].squeeze(0).to(self.device),
#                 'attention_mask': encoding['attention_mask'].squeeze(0).to(self.device)
#             },
#             'g': extra_feature.to(self.device),
#             'y': label
#         }

class BertDataset(Dataset):
    def __init__(self, df_data, tokenizer, max_length=256, extra_features=None, device=torch.device('cpu')):
        self.device = device
        self.texts = df_data['text'].tolist()
        self.labels = torch.tensor(df_data['label'].tolist(), dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

        if extra_features is not None:
            self.extra_features = extra_features
        else:
            self.extra_features = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text, padding and truncating as needed
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_tensors="pt")

        # Check if the tokenization resulted in an unexpected list or type for the length
        total_len = len(encoding['input_ids'].squeeze(0))  # Get the length of the tokenized input
        if isinstance(total_len, list):  # Ensure it's an integer, not a list
            total_len = len(total_len)  # Use the length of the list if needed

        # Now the comparison should work fine
        if total_len > self.max_length:
            print(
                f"Warning: Length of tokens exceeds max_length. Total length: {total_len}, Max length: {self.max_length}")

        # Handle extra features
        if self.extra_features is not None:
            extra_feature = self.extra_features[idx]
            extra_feature_tensor = torch.tensor(extra_feature, dtype=torch.float32).to(self.device)
        else:
            extra_feature_tensor = torch.zeros(1, dtype=torch.float32).to(self.device)

        return {
            'x': {
                'input_ids': encoding['input_ids'].squeeze(0).to(self.device),
                'token_type_ids': encoding['token_type_ids'].squeeze(0).to(self.device),
                'attention_mask': encoding['attention_mask'].squeeze(0).to(self.device),
            },
            'g': extra_feature_tensor,
            'y': label.to(self.device)
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
    自定义 trainer
'''


class CustomTrainer:
    def __init__(self, classifier_model, training_args, train_dataloader, eval_dataloader,
                 compute_metrics, device='cpu'):

        self.device = torch.device(device)
        self.classifier_model = classifier_model.to(self.device)
        # self.embedd_layer = embedd_layer.to(self.device)

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

    def run_epoch(self):
        best_val_loss = float('inf')
        best_metrics = None  # 初始化最佳指标
        early_stop_counter = 0

        for epoch in range(self.training_args['num_epochs']):
            # 1. 训练模型
            self.run_train(epoch)

            # 2. 评估模型
            val_metrics, val_loss = self.run_evaluate(eval_dataloader=self.eval_dataloader)

            # 3. 更新最佳验证损失和指标
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics.copy()  # 深拷贝当前的指标
                early_stop_counter = 0

                # 如果需要保存模型，可以在这里添加代码
                # self.save_model()

            else:
                early_stop_counter += 1

            print(val_metrics)

            if early_stop_counter >= self.training_args['early_stopping_patience']:
                print("\nEarly stopping triggered.")
                break

        return best_metrics, best_val_loss

    def run_train(self, epoch):
        # 1. 设置train mode
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
            outputs = self.classifier_model(x, g)
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
