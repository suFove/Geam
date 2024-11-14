import torch
from transformers import BertTokenizer, BertModel, AlbertTokenizerFast

from config.config import Config

config = Config()
# 加载分词器和模型
tokenizer = AlbertTokenizerFast.from_pretrained(config.bert_path)
model = BertModel.from_pretrained(config.bert_path)


# 示例文本
text = ["Hello, how are you?", "I am fine, thank you!"]

# 使用分词器对文本进行编码，返回input_ids, attention_mask, token_type_ids
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# 获取input_ids, attention_mask 和 token_type_ids
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
token_type_ids = inputs['token_type_ids']

# 获取BERT的Embedding层输出
with torch.no_grad():
    # 获取词嵌入（embedding）输出
    embedding_output = model.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

# 特征增强操作：这里简单地做一个加法操作，当然你可以根据需要做更复杂的特征增强
enhanced_embedding = embedding_output + 0.1 * torch.randn_like(embedding_output)

print(enhanced_embedding.shape)
# 将增强后的embedding送入到Encoder中
encoder_outputs = model(
    inputs_embeds=enhanced_embedding,  # 使用增强后的embedding
    attention_mask=attention_mask,     # 提供attention_mask
    token_type_ids=token_type_ids      # 提供token_type_ids（如果有多个句子）
)

# 获取Encoder的最后隐藏状态
last_hidden_state = encoder_outputs.last_hidden_state

# 输出Encoder的最后隐藏状态
print(last_hidden_state.shape)
