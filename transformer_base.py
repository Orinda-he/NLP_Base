import torch
import torch.nn as nn
import math

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# 极简 Transformer 模型
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, dim_feedforward):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        # 定义单层 Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 定义单层 Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # 嵌入和位置编码
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.positional_encoding(src)
        tgt = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt = self.positional_encoding(tgt)

        # 编码器前向传播
        memory = self.encoder(src)

        # 解码器前向传播
        output = self.decoder(tgt, memory)

        # 线性层
        output = self.fc(output)
        return output


# 示例使用
vocab_size = 1000
d_model = 512
nhead = 8
dim_feedforward = 2048

model = SimpleTransformer(vocab_size, d_model, nhead, dim_feedforward)

# 生成一些随机输入
src = torch.randint(0, vocab_size, (10, 32))  # 输入序列长度为 10，批次大小为 32
tgt = torch.randint(0, vocab_size, (10, 32))  # 目标序列长度为 10，批次大小为 32

# 前向传播
output = model(src, tgt)
print(output.shape)