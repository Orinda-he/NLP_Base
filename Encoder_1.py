"""
以下是对上述代码输出内容的详细分析及示例输出：

### 代码输出分析
代码的最后部分打印了输入张量的形状、编码器输出的形状以及编码器隐藏状态的形状，下面分别解释这些形状的含义：
1. **输入张量的形状**：输入张量经过 `unsqueeze(0)` 添加了一个批次维度，假设批次大小为 1。每个词的词向量维度为 50，示例句子分词后有 6 个词，所以输入张量的形状为 `(1, 6, 50)`，其中第一个维度表示批次大小，第二个维度表示序列长度（即词的数量），第三个维度表示词向量的维度。
2. **编码器输出的形状**：RNN 编码器的输出是每个时间步（即每个词）对应的隐藏状态。由于批次大小为 1，序列长度为 6，隐藏层维度设置为 128，所以编码器输出的形状为 `(1, 6, 128)`。
3. **编码器隐藏状态的形状**：RNN 最后一个时间步的隐藏状态，其形状为 `(1, 1, 128)`，第一个维度表示层数（这里简单 RNN 只有 1 层），第二个维度表示批次大小，第三个维度表示隐藏层维度。

### 示例输出
运行上述代码，可能会得到如下输出：
```
输入张量的形状: torch.Size([1, 6, 50])
编码器输出的形状: torch.Size([1, 6, 128])
编码器隐藏状态的形状: torch.Size([1, 1, 128])
```

这些输出结果清晰地展示了数据在 RNN 编码器中的流动和处理过程，有助于我们理解输入数据的结构以及编码器的输出形式。
"""
import torch
import torch.nn as nn
from torchtext.vocab import GloVe

# 加载预训练的GloVe词向量，这里使用维度为50的词向量
glove = GloVe(name='6B', dim=50)

# 示例句子
sentence = "This movie is really amazing!"

# 分词
tokens = sentence.split()

# 将每个词转换为对应的词向量
input_vectors = []
for token in tokens:
    vector = glove[token]
    input_vectors.append(vector)

# 将词向量列表转换为torch.Tensor
input_tensor = torch.stack(input_vectors).unsqueeze(0)
# 这里的unsqueeze(0)是为了添加一个批次维度，假设批次大小为1

# 定义一个简单的RNN编码器
class SimpleRNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleRNNEncoder, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        output, hidden = self.rnn(x)
        return output, hidden

# 初始化编码器
input_size = 50  # 词向量的维度
hidden_size = 128  # 隐藏层的维度
encoder = SimpleRNNEncoder(input_size, hidden_size)

# 输入编码器进行处理
output, hidden = encoder(input_tensor)

print("输入张量的形状:", input_tensor.shape)
print("编码器输出的形状:", output.shape)
print("编码器隐藏状态的形状:", hidden.shape)