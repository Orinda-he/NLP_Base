import torch
import torch.nn as nn
import torch.optim as optim

# 定义字符到索引和索引到字符的映射
vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
for char in 'abcdefghijklmnopqrstuvwxyz':
    vocab[char] = len(vocab)
inv_vocab = {v: k for k, v in vocab.items()}

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# 解码器
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# 辅助函数：将字符串转换为索引序列
def string_to_indices(s):
    indices = [vocab[char] for char in s]
    indices = [vocab['<SOS>']] + indices + [vocab['<EOS>']]
    return torch.tensor(indices, dtype=torch.long)

# 辅助函数：将索引序列转换为字符串
def indices_to_string(indices):
    return ''.join([inv_vocab[idx.item()] for idx in indices if idx.item() not in [vocab['<SOS>'], vocab['<EOS>'], vocab['<PAD>']]])

# 训练参数
input_size = len(vocab)
hidden_size = 128
output_size = len(vocab)
learning_rate = 0.01

# 初始化编码器和解码器
encoder = Encoder(input_size, hidden_size)
decoder = Decoder(hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.NLLLoss()
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

# 输入序列
input_string = "hello"
target_string = input_string[::-1]

input_indices = string_to_indices(input_string)
target_indices = string_to_indices(target_string)

# 编码过程
encoder_hidden = encoder.initHidden()
encoder_outputs = []
for i in range(input_indices.size(0)):
    encoder_output, encoder_hidden = encoder(input_indices[i].unsqueeze(0), encoder_hidden)
    encoder_outputs.append(encoder_output)

# 上下文向量
context = encoder_hidden

# 解码过程
decoder_input = torch.tensor([[vocab['<SOS>']]])
decoder_hidden = context
decoded_indices = []

for i in range(target_indices.size(0)):
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    topv, topi = decoder_output.topk(1)
    decoder_input = topi.squeeze().detach()
    decoded_indices.append(decoder_input)
    if decoder_input.item() == vocab['<EOS>']:
        break

# 输出结果
print("输入字符串:", input_string)
print("目标字符串:", target_string)
print("编码过程输出的最后一个隐藏状态 (上下文向量):", context.squeeze().detach().numpy())
print("解码后的字符串:", indices_to_string(decoded_indices))

