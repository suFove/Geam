import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import json


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.data.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


class CustomTrainer:
    def __init__(self, model, train_dataloader, eval_dataloader, device='cpu', output_file=None, compute_metrics=None):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=5e-5)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=len(train_dataloader) // 4,
                                                         num_training_steps=len(train_dataloader))
        self.output_file = output_file
        self.compute_metrics = compute_metrics

    def train(self):
        self.model.train()
        for batch in self.train_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

    def evaluate(self):
        self.model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)

                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                labels = batch['labels'].cpu().numpy()
                all_labels.extend(labels)

        # 计算评价指标
        metrics = {}
        if self.compute_metrics is not None:
            metrics = self.compute_metrics((all_preds, all_labels))

        print(f'Metrics: {metrics}')
        return metrics

    def save_metrics(self, metrics):
        if self.output_file is not None:
            with open(self.output_file, 'w') as f:
                json.dump(metrics, f)
            print(f'Metrics saved to {self.output_file}')


# 示例数据集和模型创建（这需要根据实际情况替换为你的实际内容）
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

data = tokenizer(["Hello world", "How are you"], padding=True)
labels = [1, 0]

dataset_dict = {
    "train": CustomDataset(data, labels),
    "dev": CustomDataset(data, labels)  # 使用相同的数据集进行简化演示
}

train_dataloader = DataLoader(dataset_dict["train"], batch_size=2)
eval_dataloader = DataLoader(dataset_dict["dev"], batch_size=2)


# 定义自定义的计算指标函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    accuracy = (predictions == labels).mean()
    return {"accuracy": float(accuracy)}


trainer = CustomTrainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_file="metrics.json",
    compute_metrics=compute_metrics
)

trainer.train()
metrics = trainer.evaluate()

trainer.save_metrics(metrics)

print(metrics)  # 输出评价指标