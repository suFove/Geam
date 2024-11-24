import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from pre.train_vector import load_vec_model
from trains.dl import init_dl_runner
from utils.class_hub import EmbeddingHandler
from utils.common import init_components, dataloader2flatten, get_base_model, compute_metrics


def run_ml():
    # 1.创造初始化组件
    config, word_idx_dict, idx_tensor_dict = init_components()
    print(f'Current dataset is: {config.dataset_name}')
    print(f"The category is: {config.dataset_info[config.dataset_name]['num_labels']}")
    all_df, train_df, dev_df, test_df, word2vec_model = load_vec_model(config)
    eh = EmbeddingHandler(config, word_idx_dict, idx_tensor_dict)

    # 2. 创建模型
    # _, feature_fusion_model = get_base_model(config)  # 忽略dl模型
    models = {
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier(),
        'LR': LogisticRegression(),
        'NB': GaussianNB(),
    }

    # 3. 创建dataloader
    train_loader, dev_loader, test_loader = init_dl_runner(config, train_df, dev_df, test_df, word2vec_model, eh)

    train_x, train_y = dataloader2flatten(train_loader)
    dev_x, dev_y = dataloader2flatten(dev_loader)
    test_x, test_y = dataloader2flatten(dev_loader)

    # 标准化
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    dev_x = scaler.transform(dev_x)
    test_x = scaler.transform(test_x)

    for model_name, model in models.items():
        model.fit(train_x, np.ravel(train_y))
        metrics_dev_report = compute_metrics(dev_y, model.predict(dev_x))
        metrics_test_report = compute_metrics(test_y, model.predict(test_x))
        print("eval", model_name, metrics_dev_report)
        print("test", model_name, metrics_test_report)


if __name__ == '__main__':
    run_ml()
