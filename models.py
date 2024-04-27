import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder
from preprocess import get_word2id, get_word2vec

word2id = get_word2id()
word2vec = get_word2vec(word2id=word2id)


class CONFIG:
    # Whether to update the word2vec model during training
    update_w2v = True

    # Vocabulary size, consistent with the word2id dictionary
    vocab_size = len(word2id) + 1

    # Number of classes: pos and neg
    n_class = 2

    # Word embedding dimension
    embedding_dim = 50

    # Number of convolutional filters
    kernel_num = 20

    # Convolutional kernel sizes
    kernel_size = [3, 5, 7]

    # Pre-trained word embedding model
    pretrained_embed = word2vec

    # Hidden layer size
    hidden_size = 256

    # Number of hidden layers
    num_layers = 2

    # Transformer model dimension
    d_model = 256

    # Number of attention heads in Transformer
    nhead = 8

    # Number of Transformer layers
    num_transformer_layers = 8

    # Transformer feed-forward layer size
    dim_feedforward = 1024

    # Dropout rate
    dropout = 0.3

    # Number of hidden layers in BERT model
    num_hidden_layers = 12

    # Number of attention heads in BERT
    num_attention_heads = 8

    # Size of intermediate layer in BERT
    intermediate_size = 384

    # Maximum position embedding length in BERT
    max_position_embeddings = 64

    # Vocabulary size for token type ids in BERT
    type_vocab_size = 2

    # Token ID for padding in BERT
    pad_token_id = 0

    # Layer normalization epsilon in BERT
    layer_norm_eps = 1e-12

    # Dropout rate for attention probabilities in BERT
    attention_probs_dropout_prob = 0.1

    # Initialization range for BERT parameters
    initializer_range = 0.02

    # Chunk size for feed-forward in BERT
    chunk_size_feed_forward = 0

    # Dropout rate for hidden layers in BERT
    hidden_dropout_prob = 0.1

    # Dropout rate for attention in BERT
    attention_dropout_prob = 0.1

    # Maximum sequence length in BERT
    max_seq_length = 64

    # Use gradient checkpointing in BERT
    gradient_checkpointing = False

    # Use fused layer normalization in BERT
    fused_layer_norm = True

    # Label smoothing factor in BERT
    label_smoothing = 0.0

    # Warm-up steps in BERT
    warmup_steps = 0

    # Weight decay in BERT
    weight_decay = 0.01

    # Adam beta1 in BERT
    adam_beta1 = 0.9

    # Adam beta2 in BERT
    adam_beta2 = 0.999

    # Adam epsilon in BERT
    adam_epsilon = 1e-8

    # Whether the model is a decoder in BERT
    is_decoder = False

    # Whether to use cross-attention in BERT
    add_cross_attention = False

    # Whether to tie word embeddings in BERT
    tie_word_embeddings = False

    # Whether to use cache in BERT
    use_cache = True

    # Whether the model is an encoder-decoder in BERT
    is_encoder_decoder = False

    # Activation function in BERT
    hidden_act = "gelu"


class TextCNN(nn.Module):
    def __init__(self, config: CONFIG):
        """
        TextCNN 模型的初始化方法。

        Args:
            config (CONFIG): 包含模型超参数的配置对象。
        """
        super(TextCNN, self).__init__()
        self.__name__ = "TextCNN"

        # 定义词嵌入层
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embedding.weight.requires_grad = config.update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embed))

        # 定义多个卷积层
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, config.kernel_num, (size, config.embedding_dim))
                for size in config.kernel_size
            ]
        )

        # 定义 Dropout 层和全连接层
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Sequential(
            self.dropout,
            nn.Linear(len(config.kernel_size) * config.kernel_num, config.n_class),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        """
        TextCNN 模型的前向传播方法。

        Args:
            x (torch.Tensor): 输入的词ID序列。

        Returns:
            torch.Tensor: 模型的分类结果。
        """
        # 将输入序列转换为词向量, 并增加通道维度
        x = self.embedding(x.to(torch.int64)).unsqueeze(1)

        # 对词向量进行卷积和池化操作
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        # 将不同卷积核得到的特征拼接起来
        x = torch.cat(x, 1)

        # 通过全连接层得到最终的分类结果
        return self.fc(x)


class RNN_LSTM(nn.Module):
    def __init__(self, config: CONFIG):
        """
        RNN-LSTM 模型的初始化方法。

        Args:
            config (CONFIG): 包含模型超参数的配置对象。
        """
        super(RNN_LSTM, self).__init__()
        self.__name__ = "RNN_LSTM"

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embedding.weight.requires_grad = config.update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embed))

        self.encoder = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.decoder = nn.Sequential(
            nn.Linear(2 * config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, config.n_class),
        )

    def forward(self, inputs):
        """
        RNN-LSTM 模型的前向传播方法。

        Args:
            inputs (torch.Tensor): 输入的词ID序列。

        Returns:
            torch.Tensor: 模型的分类结果。
        """
        embeddings = self.embedding(inputs.to(torch.int64))
        outputs, (h_n, c_n) = self.encoder(embeddings)
        # 使用双向LSTM的最后一层的隐藏状态作为特征
        features = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.decoder(features)


class RNN_GRU(nn.Module):
    def __init__(self, config: CONFIG):
        """
        RNN-GRU 模型的初始化方法。

        Args:
            config (CONFIG): 包含模型超参数的配置对象。
        """
        super(RNN_GRU, self).__init__()
        self.__name__ = "RNN_GRU"

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embedding.weight.requires_grad = config.update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embed))

        self.encoder = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.decoder = nn.Sequential(
            nn.Linear(2 * config.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, config.n_class),
        )

    def forward(self, inputs):
        """
        RNN-GRU 模型的前向传播方法。

        Args:
            inputs (torch.Tensor): 输入的词ID序列。

        Returns:
            torch.Tensor: 模型的分类结果。
        """
        embeddings = self.embedding(inputs.to(torch.int64))
        outputs, h_n = self.encoder(embeddings)
        # 使用双向GRU的最后一层的隐藏状态作为特征
        features = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.decoder(features)


class MLP(nn.Module):
    def __init__(self, config: CONFIG):
        """
        MLP 模型的初始化方法。

        Args:
            config (CONFIG): 包含模型超参数的配置对象。
        """
        super(MLP, self).__init__()
        self.__name__ = "MLP"

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embedding.weight.requires_grad = config.update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embed))

        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.n_class),
        )

        # 使用Xavier初始化
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        """
        MLP 模型的前向传播方法。

        Args:
            inputs (torch.Tensor): 输入的词ID序列。

        Returns:
            torch.Tensor: 模型的分类结果。
        """
        embeddings = self.embedding(inputs.to(torch.int64))
        output = self.mlp(embeddings.mean(dim=1))
        return output


class Transformer(nn.Module):
    def __init__(self, config: CONFIG):
        super(Transformer, self).__init__()
        self.__name__ = "Transformer"

        # 定义嵌入层
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embedding.weight.requires_grad = config.update_w2v
        self.embedding.weight.data.copy_(torch.from_numpy(config.pretrained_embed))

        # 添加一个线性层将嵌入投影到所需维度
        self.projection = nn.Linear(config.embedding_dim, config.d_model)

        # 定义位置编码
        # self.pos_encoder = PositionalEncoding(config.d_model, config.dropout)

        # 定义 Transformer 编码器层
        self.transformer_encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    config.d_model,
                    config.nhead,
                    config.dim_feedforward,
                    config.dropout,
                )
                for _ in range(config.num_transformer_layers)
            ]
        )
        self.transformer_encoder = TransformerEncoder(
            self.transformer_encoder_layers,
            config.num_transformer_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        # 定义全连接层
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.d_model, config.n_class)
        self.log_softmax = nn.LogSoftmax(dim=1)

    # 其他代码保持不变

    def forward(self, x):
        """
        Transformer 文本分类模型的前向传播方法。

        Args:
            x (torch.Tensor): 输入的词ID序列。

        Returns:
            torch.Tensor: 模型的分类结果。
        """
        # 将输入序列转换为词嵌入
        x = self.embedding(x.to(torch.int64))

        # 将嵌入投影到所需维度
        x = self.projection(x)

        # 添加位置编码
        # x = self.pos_encoder(x)

        # 将投影后的嵌入传入 Transformer 编码器
        x = x.permute(
            1, 0, 2
        )  # (batch, seq_len, embed_dim) -> (seq_len, batch, embed_dim)
        for layer in self.transformer_encoder_layers:
            x = layer(x)
        x = x[-1, :, :]  # 取最后一个时间步的输出作为文本表示

        # 将文本表示传入全连接层
        x = self.dropout(x)
        x = self.fc(x)
        x = self.log_softmax(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Bert(nn.Module):
    def __init__(self, config: CONFIG):
        super(Bert, self).__init__()
        self.__name__ = "Bert"

        # 定义 BERT 词嵌入层
        self.embeddings = BertEmbeddings(config)

        # 定义 BERT 编码器层
        self.bert_encoder = BertEncoder(config)

        # 定义 BERT 分类头
        self.dropout = nn.Dropout(config.dropout)
        self.pre_classifier = nn.Linear(config.d_model, 256)
        self.classifier = nn.Linear(256, config.n_class)

        # 初始化 BERT 模型权重
        self.init_weights(config=config)

    def init_weights(self, config: CONFIG):
        # 初始化分类头的权重
        self.classifier.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.classifier.bias.data.zero_()

        # 初始化 BERT 模型其他部分的权重
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        BERT 文本分类模型的前向传播方法。

        Args:
            input_ids (torch.Tensor): 输入的词ID序列。
            attention_mask (torch.Tensor, optional): 注意力掩码。
            token_type_ids (torch.Tensor, optional): 段ID序列。

        Returns:
            torch.Tensor: 模型的分类结果。
        """
        input_ids = input_ids.long()

        # 通过 BERT 词嵌入层
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # 通过 BERT 编码器层
        encoder_outputs = self.bert_encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs[0]

        # 取 [CLS] 标记的输出作为文本表示
        pooled_output = sequence_output[:, 0, :]

        # 通过 pre_classifier 层
        pooled_output = self.pre_classifier(pooled_output)

        # 通过全连接层得到分类结果
        logits = self.classifier(self.dropout(pooled_output))

        return logits
