import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 假设 input_x 是形状为 [batch_size, max_len, embedding_dim] 的数组
input_x = np.random.rand(100, 50, 300)
label_y = np.random.randint(0, 2, (100, 1))

# 展平特征，将输入展平成一个向量，形状为 [batch_size, max_len * embedding_dim]
input_x_flattened = input_x.reshape(input_x.shape[0], -1)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(input_x_flattened, label_y, test_size=0.2, random_state=42)

# 数据预处理（标准化）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义模型
models = {
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'LR': LogisticRegression(),
    'NB': GaussianNB(),
}

# 训练和评估模型
for name, model in models.items():
    model.fit(X_train_scaled, y_train.ravel())
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy:.4f}')