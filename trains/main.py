from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BertModel
from config.config import Config
from trains.models import TextGraphFusionModule
from utils.berts import compute_metrics, CustomTrainer
from utils.common import create_dataloader, get_base_model

'''
    the data dict should be created by ../pre/make_dataset.py
'''


def init_components():
    config = Config()
    print(config.dataset_info[config.dataset_name]['num_labels'])

    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    bert_model = AutoModelForSequenceClassification.from_pretrained(config.bert_path,
                                                                    num_labels=config.dataset_info[config.dataset_name][
                                                                        'num_labels'])
    embedding_layer = BertModel(bert_model.config).embeddings
    feature_fusion_model = TextGraphFusionModule()
    classifier_model = get_base_model(config)
    classifier_model = classifier_model if classifier_model is not None else bert_model

    dataset_dict = load_from_disk(config.dataset_info[config.dataset_name]['root_path'])
    print(dataset_dict)
    print(dataset_dict['train'][0]['input_ids'])
    trainloader, devloader, testloader = create_dataloader(dataset_dict,
                                                           tokenizer,
                                                           embedding_layer,
                                                           config.training_settings['batch_size'])

    trainer = CustomTrainer(classifier_model=classifier_model,
                            training_args=config['training_settings'],
                            train_dataloader=trainloader,
                            eval_dataloader=devloader,
                            compute_metrics=compute_metrics,
                            feature_fusion_model=feature_fusion_model,
                            device=config.device)
    return trainer


def run():
    trainer = init_components()
    metrics, best_val_loss = trainer.train()
    print(f"Best Validation Loss: {best_val_loss}")
    trainer.save_metrics(config.classifier_model_name, metrics)

if __name__ == "__main__":
    components = init_components()
    run(components['config'], components['dataset_dict'], components['trainer'])
