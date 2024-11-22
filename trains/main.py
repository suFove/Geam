from transformers import BertTokenizer, AutoModelForSequenceClassification, BertModel
from pre.make_data_embedding import init_components, word_to_feature, create_dataloaders
from utils.berts import compute_metrics, CustomTrainer
from utils.common import get_base_model
'''
    the data dict should be created by ../pre/make_dataset.py
    before training the model
'''


def init_runner():
    # 1.创造初始化组件
    config, word_idx_dict, idx_tensor_dict = init_components()
    print(f'Current dataset is: {config.dataset_name}')
    print(f"The category is: {config.dataset_info[config.dataset_name]['num_labels']}")

    # 2. 创建模型
    # tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(config.bert_path,
                                                                    num_labels=config.dataset_info[config.dataset_name][
                                                                        'num_labels'])
    embedding_layer = BertModel(bert_model.config).embeddings
    classifier_model, feature_fusion_model = get_base_model(config)
    classifier_model = classifier_model if classifier_model is not None else bert_model
    print(classifier_model)
    print(feature_fusion_model)
    print("Loading dataset from disk")

    # 3.创建数据集，转为dataloader
    train_x, train_y, dev_x, dev_y, test_x, test_y, train_g = word_to_feature(config, word_idx_dict, idx_tensor_dict)
    train_loader, dev_loader, test_loader = create_dataloaders(train_x, train_y, dev_x, dev_y, test_x, test_y, train_g,
                                                               batch_size=config.training_settings['batch_size'])

    print("Loading finished")

    # 4.创建训练器
    print(f'Training on {config.device}')

    trainer = CustomTrainer(classifier_model=classifier_model,
                            training_args=config.training_settings,
                            train_dataloader=train_loader,
                            eval_dataloader=dev_loader,
                            compute_metrics=compute_metrics,
                            feature_fusion_model=feature_fusion_model,
                            device=config.device)
    return config, trainer, train_loader, dev_loader, test_loader


def run():
    config, trainer, train_loader, dev_loader, test_loader = init_runner()
    val_metrics, best_val_loss = trainer.run_epoch()
    print(f"Best Validation Loss: {best_val_loss}")
    print(f"Best Validation Metrics: {val_metrics}")
    trainer.save_metrics('val_metrics', val_metrics)

    test_metrics, best_test_loss = trainer.run_evaluate(test_loader)
    print(f"Best Test Loss: {best_val_loss}")
    print(f"Best Test Metrics: {test_metrics}")
    trainer.save_metrics('test_metrics', test_metrics)


if __name__ == "__main__":
    run()
