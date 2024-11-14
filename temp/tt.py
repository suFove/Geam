
import torch
from transformers import BertModel

class EmbeddingHandler:
    def __init__(self, model_name="../BETTER/ZY-BERT"):
        self.embedd_layer = BertModel.from_pretrained(model_name)
    def get_embeddings(self, input_ids, token_type_ids=None):
        with torch.no_grad():
            embeddings = self.embedd_layer(input_ids=input_ids.unsqueeze(0),
                                           token_type_ids=token_type_ids.unsqueeze(0) if token_type_ids is not None else None)
            return embeddings.last_hidden_state.squeeze(0)

# 使用示例
embedding_handler = EmbeddingHandler(model_name="../BERT/ZY-BERT")

input_ids = torch.tensor([[101, 2053, 1996, 18874, 4248, 1012, 1996, 9748, 102]])  # 示例输入ID
token_type_ids = torch.tensor([[0] * input_ids.size(1)])  # 如果需要，设置分段ID

embeddings = embedding_handler.get_embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
print(embeddings.shape)  # 输出应该是 [max_len, hidden_size]