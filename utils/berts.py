import json
import os
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.optim import AdamW, Adam
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

'''
============= 编码数据集 ================
'''


def compute_metrics(y_true, y_pred):
    # 计算准确率
    acc = accuracy_score(y_true, y_pred)

    # 计算精确率、召回率和F1分数，average='weighted'表示使用加权平均
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro',
                                                               zero_division=1)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


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
            print(val_metrics)
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
            x = batch['x'].to(self.device)
            g = batch['g'].to(self.device) if 'g' in batch else None
            y = batch['y'].to(self.device)
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
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)

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
