from pre.train_vector import load_vec_model
from trains.bert import init_bert_components
from trains.dl import init_dl_runner
from utils.class_hub import EmbeddingHandler, CustomTrainer
from utils.common import init_components, get_base_model, compute_metrics


def run():
    # 1.创造初始化组件
    config, word_idx_dict, idx_tensor_dict = init_components()
    print(f'Current dataset is: {config.dataset_name}')
    print(f"The category is: {config.dataset_info[config.dataset_name]['num_labels']}")
    all_df, train_df, dev_df, test_df, word2vec_model = load_vec_model(config)
    eh = EmbeddingHandler(config, word_idx_dict, idx_tensor_dict)

    # 2. 创建模型
    classifier_model, feature_fusion_model = get_base_model(config)
    print('classifier_model:', classifier_model)
    print('feature_fusion_model', feature_fusion_model)
    print("Loading dataset from disk")

    # 3. 根据模型类选择数据集加载：采用bert还是word2vec, 二者创建dataloader时，所需dataset类不同，一个返回字典，一个返回tensor
    train_loader, dev_loader, test_loader = None, None, None
    if config.classifier_model_name == 'Bert':
        train_loader, dev_loader, test_loader = init_bert_components(config, train_df, dev_df, test_df, eh)
    else:
        train_loader, dev_loader, test_loader = init_dl_runner(config, train_df, dev_df, test_df, word2vec_model, eh)

    # 4.创建训练器
    print(f'Training on {config.device}')
    trainer = CustomTrainer(classifier_model=classifier_model,
                            training_args=config.training_settings,
                            train_dataloader=train_loader,
                            eval_dataloader=dev_loader,
                            compute_metrics=compute_metrics,
                            feature_fusion_model=feature_fusion_model,
                            device=config.device)

    # 5.评价并保存指标
    val_metrics, best_val_loss = trainer.run_epoch()
    print(f"Best Validation Loss: {best_val_loss}")
    print(f"Best Validation Metrics: {val_metrics}")
    trainer.save_metrics('val_metrics', val_metrics)

    test_metrics, best_test_loss = trainer.run_evaluate(test_loader)
    print(f"Best Test Loss: {best_val_loss}")
    print(f"Best Test Metrics: {test_metrics}")
    trainer.save_metrics('test_metrics', test_metrics)


if __name__ == '__main__':
    run()
