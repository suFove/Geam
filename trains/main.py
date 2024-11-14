from datasets import load_from_disk
from transformers import BertTokenizer, AutoModelForSequenceClassification, BertModel
from config.config import Config
from trains.models import TextGraphFusionModule
from utils.berts import compute_metrics, CustomTrainer
from utils.common import create_dataloader, get_base_model

'''
    the data dict should be created by ../pre/make_dataset.py
    before training the model
'''


def init_components():
    config = Config()
    print(f'Current dataset is: {config.dataset_name}')
    print(f"The category is: {config.dataset_info[config.dataset_name]['num_labels']}")

    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(config.bert_path,
                                                                    num_labels=config.dataset_info[config.dataset_name][
                                                                        'num_labels'])
    embedding_layer = BertModel(bert_model.config).embeddings
    feature_fusion_model = TextGraphFusionModule()
    feature_fusion_model = None
    classifier_model = get_base_model(config)
    classifier_model = classifier_model if classifier_model is not None else bert_model

    print("Loading dataset from disk")
    dataset_dict = load_from_disk(config.dataset_info[config.dataset_name]['root_path'])

    train_loader, dev_loader, test_loader = create_dataloader(dataset_dict,
                                                              tokenizer,
                                                              embedding_layer,
                                                              config.training_settings['batch_size'])
    print("Loading finished")
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
    config, trainer, train_loader, dev_loader, test_loader = init_components()
    val_metrics, best_val_loss = trainer.train()
    print(f"Best Validation Loss: {best_val_loss}")
    trainer.save_metrics(config.classifier_model_name + 'val_metrics', val_metrics)

    test_metrics, best_test_loss = trainer.evaluate(test_loader)
    print(f"Best Test Loss: {best_val_loss}")
    trainer.save_metrics(config.classifier_model_name + 'test_metrics', test_metrics)


if __name__ == "__main__":
    run()
