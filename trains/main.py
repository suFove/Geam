from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BertModel
from config.config import Config
from utils.berts import compute_metrics
from utils.common import create_dataloader, get_base_model


def init_components():
    config = Config()
    print(config.dataset_info[config.dataset_name]['num_labels'])

    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(config.bert_path,
                                                               num_labels=config.dataset_info[config.dataset_name]['num_labels'])
    embedding_layer = BertModel(bert_model.config).embeddings

    classifier_model =  get_base_model(config)
    classifier_model = classifier_model if classifier_model is not None else bert_model


    dataset_dict = load_from_disk(config.dataset_info[config.dataset_name]['root_path'])
    print(dataset_dict)
    print(dataset_dict['train'][0]['input_ids'])
    trainloader, devloader, testloader = create_dataloader(dataset_dict,
                                                           tokenizer,
                                                           embedding_layer,
                                                           config.training_settings['batch_size'])

    for batch in trainloader:
        print(batch)
        print(batch['xembeddings'].shape)
        print(batch['gembeddings'].shape)

        break

    training_args = TrainingArguments(
        output_dir=config.result_dir,
        eval_strategy="epoch",
        learning_rate=config.training_settings['learning_rate'],
        per_device_train_batch_size=config.training_settings['batch_size'],
        per_device_eval_batch_size=config.training_settings['batch_size'],
        num_train_epochs=config.training_settings['epoch'],
        weight_decay=0.01,
    )

    # 初始化自定义的 Trainer 对象
    trainer = Trainer(
        model=classifier_model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["dev"],  # 使用验证集进行评估
        compute_metrics=compute_metrics,  # 使用自定义的 compute_metrics 函数
    )

    return {
        "config": config,
        "tokenizer": tokenizer,
        "model": classifier_model,
        "dataset_dict": dataset_dict,
        "trainloader": trainloader,
        "devloader": devloader,
        "testloader": testloader,
        "trainer": trainer
    }


def run(config, dataset_dict, trainer):
    trainer.train()
    # 在测试集上进行评估并记录结果
    test_results = trainer.evaluate(dataset_dict["test"], metric_key_prefix="test")
    print(test_results)
    # 打印自定义状态信息
    print(trainer.state)
    trainer.save_model(config.result_dir + f'{config.classifier_model_name}/best')
    trainer.state.save_to_json(config.result_dir + f'{config.classifier_model_name}/res.json')



if __name__ == "__main__":
    components = init_components()
    run(components['config'], components['dataset_dict'], components['trainer'])