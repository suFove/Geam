# Text Classification Repository Based on PyTorch

This repository describes how to build training processes using common machine learning (ML) algorithms, deep learning (DL) models, and transformers (BERT) for text classification tasks. It includes personal summaries and insights.

## Author Information:
- **Name:** [suFove]
- **Email:** [164206055@qq.com]

## Process Overview:
1. **Data Preprocessing**
2. **Text to Vector Conversion**

    - **Route 1 (ML&DL)**: Use jieba for Chinese word segmentation, train a Word2Vec model on the segmented tokens, and map them to vector lists. Efficient mapping can be done using `df.apply`.
    
    - **Route 2 (Transformer)**: Utilize BERT's tokenizer to encode the text into features such as `input_ids`, `token_type_ids`, and `attention_mask`.

3. **Create Dataset Extending torch.utils.data.Dataset**

    - **Route 1 (ML&DL)**: Directly manipulate the mapped vector lists, convert them to tensors (`list(torch.tensor)`), and return specific tensors using `get_item`.
    
    - **Route 2 (Transformer)**: Pass the tokenizer into the dataset for mapping text lists. The `get_item` function returns a dictionary.

4. **Create DataLoader with Respective Dataset**

5. **Training Class Creation and Training**

6. **Model Evaluation and Saving Metrics**


## Requirements
- **Environment:** 
    - pytorch=2.3.1 + cuda=11.8 + python=3.12

## Usage Steps:
1. Run `/pre/pre_data.py`
2. Execute `/pre/train_vector.py`
3. Verify configurations in `/config/config.py` to ensure the model settings are correct.
4. Run `/trains/main.py`

---

# 基于PyTorch的中文文本分类仓库

这个仓库描述了如何使用常用的机器学习(ML)算法、深度学习(DL)模型以及transformers(BERT)构建中文文本分类任务的训练过程。包括个人总结和见解。

## 作者信息：
- **姓名:** [suFove]
- **邮箱:** [164206055@qq.com]

## 流程简介：
1. **数据预处理**
2. **文本转为向量**

    - **路线1 (ML&DL)**：使用jieba进行中文分词，对分词后的token使用Word2Vec模型进行训练，并将其映射到向量列表。可以利用`df.apply`进行高效的批量映射。
    
    - **路线2 (Transformer)**：使用BERT的tokenizer对文本编码，得到特征如 `input_ids`, `token_type_ids`, 和 `attention_mask`。

3. **创建继承自 torch.utils.data.Dataset 的 Dataset**

    - **路线1 (ML&DL)**：直接操作映射后的向量列表，并将其转换为张量（使用 `list(torch.tensor)`），在 `get_item` 方法中返回具体的张量。
    
    - **路线2 (Transformer)**：将tokenizer传入dataset，对文本列表进行映射。`get_item` 返回一个字典。

4. **创建对应的数据加载器DataLoader**

5. **训练类的创建和模型训练**

6. **评估模型并保存指标**


## 运行环境
- pytorch=2.3.1 + cuda=11.8 + python=3.12

## 使用步骤：
1. 执行 `/pre/pre_data.py`
2. 执行 `/pre/train_vector.py`
3. 检查并确认配置文件`/config/config.py`中的模型设置是否正确。
4. 运行 `/trains/main.py`

---

希望这能帮助你更好地理解和使用这个仓库。如果需要进一步的细节或改进，欢迎继续讨论和修改！