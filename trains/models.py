from torch import nn
from transformers import BertModel, AutoModelForSequenceClassification, AutoModel
import torch
import numpy as np
import pandas as pd
from tqdm import trange

from utils.util4ge import feature_calculator, read_graph, adjacency_opposite_calculator
import torch.nn.functional as F

"""AttentionWalk class."""


class GraphEmbeddingLayer(torch.nn.Module):
    """
    Attention Walk Layer.
    For details see the paper.
    """

    def __init__(self, embedding_dim, num_walks, beta, gamma, shapes):
        """
        Setting up the layer.
        :param shapes: Shape of the target tensor.
        """
        super(GraphEmbeddingLayer, self).__init__()

        self.attention_probs = None
        self.embedding_dim = embedding_dim
        self.num_walks = num_walks
        self.beta = beta
        self.gamma = gamma
        self.shapes = shapes

        self.define_weights()
        self.initialize_weights()

    def define_weights(self):
        """
        Define the model weights.
        """
        half_dim = int(self.embedding_dim / 2)
        self.left_factors = torch.nn.Parameter(torch.Tensor(self.shapes[1], half_dim))
        self.right_factors = torch.nn.Parameter(torch.Tensor(half_dim, self.shapes[1]))
        self.attention = torch.nn.Parameter(torch.Tensor(self.shapes[0], 1))

    def initialize_weights(self):
        """
        Initializing the weights.
        """
        torch.nn.init.uniform_(self.left_factors, -0.01, 0.01)
        torch.nn.init.uniform_(self.right_factors, -0.01, 0.01)
        torch.nn.init.uniform_(self.attention, -0.01, 0.01)

    def forward(self, weighted_target_tensor, adjacency_opposite):
        """
        Doing a forward propagation pass.
        :param weighted_target_tensor: Target tensor factorized.
        :param adjacency_opposite: No-edge indicator matrix.
        :return loss: Loss being minimized.
        """
        self.attention_probs = torch.nn.functional.softmax(self.attention, dim=0)
        probs = self.attention_probs.unsqueeze(1).expand_as(weighted_target_tensor)
        weighted_target_tensor = weighted_target_tensor * probs
        weighted_tar_mat = torch.sum(weighted_target_tensor, dim=0)
        weighted_tar_mat = weighted_tar_mat.view(self.shapes[1], self.shapes[2])
        estimate = torch.mm(self.left_factors, self.right_factors)
        loss_on_target = - weighted_tar_mat * torch.log(torch.sigmoid(estimate))
        loss_opposite = -adjacency_opposite * torch.log(1 - torch.sigmoid(estimate))
        loss_on_mat = self.num_walks * weighted_tar_mat.shape[0] * loss_on_target + loss_opposite
        abs_loss_on_mat = torch.abs(loss_on_mat)
        average_loss_on_mat = torch.mean(abs_loss_on_mat)
        norms = torch.mean(torch.abs(self.left_factors)) + torch.mean(torch.abs(self.right_factors))
        loss_on_regularization = self.beta * (self.attention.norm(2) ** 2)
        loss = average_loss_on_mat + loss_on_regularization + self.gamma * norms
        return loss


"""
    graph embedding

"""


class GraphEmbeddingTrainer(object):
    """
    Class for training the AttentionWalk model.
    """

    def __init__(self, args: dict, initialize=False):
        """
        Initializing the training object.
        :param args: Arguments object.
        """
        self.target_tensor = None
        self.adjacency_opposite = None
        self.model = None
        self.args = args
        if initialize:
            self.graph = read_graph(self.args['graph_idx_path'])
            self.initialize_model_and_features()

    def initialize_model_and_features(self):
        """
        Creating data tensors and factroization model.
        """
        self.target_tensor = feature_calculator(self.graph, window_size=self.args['window_size'])
        self.target_tensor = torch.FloatTensor(self.target_tensor)
        self.adjacency_opposite = adjacency_opposite_calculator(self.graph)
        self.adjacency_opposite = torch.FloatTensor(self.adjacency_opposite)
        self.model = GraphEmbeddingLayer(embedding_dim=self.args['embedding_dim'],
                                         num_walks=self.args['num_walks'],
                                         beta=self.args['beta'],
                                         gamma=self.args['gamma'],
                                         shapes=self.target_tensor.shape)

    def fit(self):
        """
        Fitting the model
        """
        print("\nTraining the model.\n")
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args['learning_rate'])
        epoch = trange(self.args['epoch'], desc="Loss")
        for _ in epoch:
            optimizer.zero_grad()
            loss = self.model(self.target_tensor, self.adjacency_opposite)
            loss.backward()
            optimizer.step()
            epoch.set_description("Graph Embedding (Loss=%g)" % round(loss.item(), 4))

    def save_model(self):
        """
        Saving the embedding and attention vector.
        """
        self.save_embedding()
        self.save_attention()

    def build_embedding(self):
        left = self.model.left_factors.detach().numpy()
        right = self.model.right_factors.detach().numpy().T
        indices = np.array([range(len(self.graph))]).reshape(-1, 1)
        embedding = np.concatenate([indices, left, right], axis=1)
        columns = ["id"] + ["x_" + str(x) for x in range(self.args['embedding_dim'])]
        embedding = pd.DataFrame(embedding, columns=columns)
        return embedding

    def save_embedding(self):
        """
        Saving the embedding matrices as one unified embedding.
        """
        print("\nSaving the model.\n")
        embedding = self.build_embedding()
        embedding.to_csv(self.args['graph_embedding_path'], index=None)

    def save_attention(self):
        """
        Saving the attention vector.
        """
        attention = self.model.attention_probs.detach().numpy()
        indices = np.array([range(self.args['window_size'])]).reshape(-1, 1)
        attention = np.concatenate([indices, attention], axis=1)
        attention = pd.DataFrame(attention, columns=["Order", "Weight"])
        attention.to_csv(self.args['attention_path'], index=None)


'''
   ============= 特征融合模型 =============
'''


class TextGraphFusionModule(nn.Module):
    def __init__(self, ):
        super(TextGraphFusionModule, self).__init__()
        # 两个卷积层用于处理channel维度上的特征融合
        self.avg_conv = nn.Conv2d(1, 1, kernel_size=1, padding=0)
        self.max_conv = nn.Conv2d(1, 1, kernel_size=1, padding=0)

        # padding = kernel_size // 2
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, text_feature, graph_feature):
        # step 1: 数据预处理 【batch, channel, seq_len, embedding_dim】，拓展channel维度
        text_feature = text_feature.unsqueeze(1)
        graph_feature = graph_feature.unsqueeze(1)

        # batch, channel, seq_len, embedding_dim
        b, c, s, e = text_feature.shape

        # Step 2: 计算文本和图嵌入的交叉注意力
        text_features_flattened = text_feature.view(b, c, -1)  # [b, c, h*w]
        graph_embeddings_flattened = text_feature.view(b, c, -1)

        # channel attention, [b, c, (s*e)] -> [b, c, 1]
        tf_c_att = self.channel_attention(text_features_flattened)
        gf_c_att = self.channel_attention(graph_embeddings_flattened)

        cross_att = torch.matmul(tf_c_att, gf_c_att.permute(0, 2, 1))  # batch不变，后2维交换

        tf_cross_weighted = torch.matmul(F.softmax(cross_att, dim=-1), tf_c_att)
        gf_cross_weighted = torch.matmul(F.softmax(cross_att, dim=1), gf_c_att)

        # Step 3: 空间注意力, [b, c, s, e] -> [b, c, s, e]
        tf_s_att = self.spatial_attention(tf_cross_weighted, c)
        gf_s_att = self.spatial_attention(gf_cross_weighted, c)

        # step 4: 归一化
        norm_tf = torch.softmax(tf_s_att, dim=2)
        norm_gf = torch.softmax(gf_s_att, dim=2)

        # Step 4: 特征融合
        tf_fused = text_feature + norm_tf * text_feature
        gf_fused = graph_feature + norm_gf * graph_feature

        return tf_fused.view(b, s, e) + gf_fused.view(b, s, e)

    '''
        input [batch_size, channel, (seq_len * embedding_dim)]
        output [b, c, 1]
    '''

    def channel_attention(self, input_features):
        # 求 avg & max
        avg_f = torch.mean(input_features, dim=-1, keepdim=True).unsqueeze(-1)
        max_f, _ = torch.max(input_features, dim=-1, keepdim=True)  # [b, c, 1]
        max_f = max_f.unsqueeze(-1)  # [b, c, 1, 1]

        # 分别通过avg, max卷积
        avg_att = F.relu(self.avg_conv(avg_f)).squeeze(-1)  # [b, c, 1, 1]
        max_att = F.relu(self.max_conv(max_f)).squeeze(-1)  # [b, c, 1, 1]
        att_fusion = avg_att + max_att
        return att_fusion  # [b, c, 1]

    '''
        input [b, c, s, e]
        -> [b, 2c, s, e]
        output [b, c, s, e]
    '''

    def spatial_attention(self, input_feature, channel_size=1):
        # 如果input是 channel size = 1, 则进行通道复制
        if channel_size == 1:
            input_feature = input_feature.repeat(1, 2, 1, 1)
        avg_out = torch.mean(input_feature, dim=1, keepdim=True)
        max_out, _ = torch.max(input_feature, dim=1, keepdim=True)

        # 拼接结果
        concat_f = torch.cat([avg_out, max_out], dim=1)
        att_cat = F.relu(self.spatial_conv(concat_f))
        return att_cat


'''
   ============= 基础分类模型 =============
'''


class TextCNN(nn.Module):
    def __init__(self, embed_dim, num_labels, num_filters, filter_sizes):
        super(TextCNN, self).__init__()
        # Construct convolutional layers
        # self.convs = nn.ModuleList([
        #     nn.Conv2d(1, num_filters, (fs, embed_dim), bias=True)  # Keep float32 type
        #     for fs in filter_sizes
        # ])
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, num_filters, (fs, embed_dim), bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU()
            ) for fs in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_labels)

    def forward(self, x):

        x = x.unsqueeze(1)  # Add a dimension and make sure the input is float32
        # Convolution operation followed by ReLU and max pooling
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)  # Concatenate the outputs
        # Fully connected layer
        pred_x = self.fc(x)
        return pred_x


class BiGRU_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BiGRU_Attention, self).__init__()
        self.bigru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        output, _ = self.bigru(x)

        attention_weights = torch.softmax(self.attention(output), dim=1)
        context_vector = torch.sum(output * attention_weights, dim=1)

        output = self.fc(context_vector)

        return output

class BiGRU(nn.Module):
    def __init__(self, embed_dim, num_labels, hidden_dim, num_layers=1, dropout=0.5):
        super(BiGRU, self).__init__()
        # 双向 GRU 层
        self.bigru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        # 全连接层，用于分类
        self.fc = nn.Linear(hidden_dim * 2, num_labels)  # 因为是双向的，所以隐藏状态维度要乘以2

    def forward(self, x):
        # 双向 GRU 层操作
        _, h_n = self.bigru(x)  # h_n 形状是 (num_layers * num_directions, batch_size, hidden_dim)
        # 将前向和后向的隐藏状态拼接在一起，得到最终的上下文表示
        forward_hidden = h_n[-2]  # 前向 GRU 的最后一个时间步的输出
        backward_hidden = h_n[-1]  # 后向 GRU 的最后一个时间步的输出
        context_vector = torch.cat((forward_hidden, backward_hidden), dim=1)  # (batch_size, hidden_dim * 2)
        # 全连接层操作
        out = self.fc(context_vector)
        return out


# 定义一个新的模型实例来处理融合后的特征
class ClassifierBERT(torch.nn.Module):
    def __init__(self, config, num_labels):
        super(ClassifierBERT, self).__init__()
        # 使用AutoModel加载预训练模型，不包含分类层
        self.bert = AutoModel.from_pretrained(config.bert_path)
        # 定义一个线性层用于分类
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, x):
        # 假设x是形状为[batch_size, seq_len, embedding_dim]的嵌入表示
        outputs = self.bert(inputs_embeds=x)
        pooled_output = outputs.pooler_output  # 获取 CLS token 的输出
        logits = self.fc(pooled_output)

        return logits