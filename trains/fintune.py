from datasets import load_dataset
from transformers import TrainingArguments, AlbertTokenizerFast, BertForSequenceClassification, \
    AutoModelForSequenceClassification, AlbertForSequenceClassification, Trainer
from config.config import Config
from utils.berts import compute_metrics

config = Config()

print(config.dataset_info[config.dataset_name]['num_labels'])

data_files = {"train": config.dataset_info[config.dataset_name]['train_path'],
              "dev": config.dataset_info[config.dataset_name]['dev_path'],
              "test": config.dataset_info[config.dataset_name]['test_path']}
dataset = load_dataset("csv", data_files=data_files)

# print(dataset)

tokenizer = AlbertTokenizerFast.from_pretrained(config.bert_path)
model = AlbertForSequenceClassification.from_pretrained(config.bert_path,
                                                        num_labels=config.dataset_info[config.dataset_name][
                                                            'num_labels'])


def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length',
                     max_length=config.training_settings['max_seq_len'])


encoded_dataset = dataset.map(preprocess_function, batched=True)

# print(encoded_dataset)
# print(encoded_dataset['dev']['input_ids'][0])


training_args = TrainingArguments(
    output_dir="../results/Albert",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 初始化自定义的 Trainer 对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["dev"],  # 使用验证集进行评估
    compute_metrics=compute_metrics,  # 使用自定义的 compute_metrics 函数

)

trainer.train()

# 在测试集上进行评估并记录结果
test_results = trainer.evaluate(encoded_dataset["test"], metric_key_prefix="test")
# 打印自定义状态信息
print(trainer.state)
trainer.save_model('../results/Albert/best')
trainer.state.save_to_json('../results/Albert/best/res.json')
