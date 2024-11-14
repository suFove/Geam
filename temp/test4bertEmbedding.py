import torch
from transformers import BertModel, AlbertTokenizerFast

from config.config import Config

config = Config()
# 加载分词器和模型
tokenizer = AlbertTokenizerFast.from_pretrained(config.bert_path)
model = BertModel.from_pretrained(config.bert_path)

# 获取嵌入层
embedding_layer = model.embeddings
encoder = model.encoder  # 获取编码器部分

text_list = ["Hello world", "This is a test sentence.", "Another example"]
inputs = tokenizer(text_list, max_length=12, padding='max_length', truncation=True, return_tensors='pt')

input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids']
attention_mask = inputs['attention_mask']

# 生成嵌入表示
embeddings = embedding_layer(input_ids, token_type_ids=token_type_ids)

# 自定义特征融合操作 (示例)
additional_features = torch.randn_like(embeddings) * 0.1
fused_embeddings = embeddings + additional_features

print(fused_embeddings.shape)

encoder_outputs = model.encoder(additional_features,  attention_mask=attention_mask.bool())
last_hidden_state = encoder_outputs.last_hidden_state

print(last_hidden_state.shape)  # 输出形状 (batch_size, sequence_length, hidden_dim)

#
# # 将注意力掩码转换为布尔型
# attn_mask = attention_mask.float()
# attn_mask = attn_mask.unsqueeze(-1).expand_as(fused_embeddings)
# # attn_mask_bool = attention_mask.unsqueeze(-1).expand_as(fused_embeddings)
# print(fused_embeddings.shape)
# print(attn_mask.shape)
# # 将融合后的嵌入特征传递给编码器
# encoder_outputs = encoder(fused_embeddings, attention_mask=attn_mask)
#
# # 获取最后的隐藏状态（每个token的嵌入表示）
# last_hidden_states = encoder_outputs.last_hidden_state
# print(last_hidden_states.shape)  # 输出形状应为 (batch_size, seq_length, hidden_size)

#
# # 使用 scaled_dot_product_attention 函数
# query = fused_embeddings  # 示例查询张量
# key = fused_embeddings    # 示例键张量
# value = fused_embeddings  # 示例值张量
#
# attn_output = torch.nn.functional.scaled_dot_product_attention(
#     query=query,
#     key=key,
#     value=value,
#     attn_mask=attn_mask_bool,  # 使用布尔型掩码
#     dropout_p=0.1  # 可选参数，指定dropout概率
# )
#
# print(attn_output.shape)