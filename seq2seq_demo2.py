import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 示例数据
input_sentences = ['hello world', 'how are you', 'i love machine learning','peace']
output_sentences = ['你好，世界', '你好吗', '我喜欢机器学习','心安']

# 预处理
def tokenize(sentences):
    tokenized = []
    for sentence in sentences:
        tokens = list(sentence)
        tokenized.append(tokens)
    return tokenized

input_tokenized = tokenize(input_sentences)
output_tokenized = tokenize(output_sentences)
print(input_tokenized)
print(output_tokenized)

# 构建词汇表
def build_vocab(tokenized_sentences):
    vocab = set()
    for sentence in tokenized_sentences:
        for token in sentence:
            vocab.add(token)
    vocab.add('<SOS>')  # 起始符
    vocab.add('<EOS>')  # 结束符
    return sorted(vocab)

input_vocab = build_vocab(input_tokenized)
output_vocab = build_vocab(output_tokenized)
print(input_vocab)
print(output_vocab)

input_word2idx = {word: idx for idx, word in enumerate(input_vocab)}
output_word2idx = {word: idx for idx, word in enumerate(output_vocab)}

print(input_word2idx)
print(output_word2idx)


input_idx2word = {idx: word for idx, word in enumerate(input_vocab)}
output_idx2word = {idx: word for idx, word in enumerate(output_vocab)}
print(input_idx2word )
print(output_idx2word)

# 参数设置
max_input_len = max(len(sentence) for sentence in input_tokenized)
max_output_len = max(len(sentence) for sentence in output_tokenized) + 2  # 加上 <SOS> 和 <EOS>
input_vocab_size = len(input_vocab)
output_vocab_size = len(output_vocab)
embedding_dim = 10
hidden_dim = 20

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

# 解码器
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

# Seq2Seq 模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)

        hidden, cell = self.encoder(src)

        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs

# 初始化模型、损失函数和优化器
encoder = Encoder(input_vocab_size, embedding_dim, hidden_dim)
decoder = Decoder(output_vocab_size, embedding_dim, hidden_dim)
model = Seq2Seq(encoder, decoder)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 数据处理
def sentence_to_tensor(sentence, word2idx, max_len):
    tokens = list(sentence)
    indices = [word2idx[token] for token in tokens]
    indices = [word2idx['<SOS>']] + indices + [word2idx['<EOS>']]
    indices += [0] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long).unsqueeze(1)

input_tensors = [sentence_to_tensor(sentence, input_word2idx, max_input_len) for sentence in input_sentences]
output_tensors = [sentence_to_tensor(sentence, output_word2idx, max_output_len) for sentence in output_sentences]

# 训练模型
n_epochs = 500
for epoch in range(n_epochs):
    optimizer.zero_grad()
    total_loss = 0
    for src, trg in zip(input_tensors, output_tensors):
        output = model(src, trg)
        output = output[1:].view(-1, output_vocab_size)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        total_loss += loss.item()
        loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(input_sentences)}')

# 预测
def translate_sentence(sentence, model, input_word2idx, output_idx2word, max_input_len, max_output_len):
    model.eval()
    src_tensor = sentence_to_tensor(sentence, input_word2idx, max_input_len)
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
    trg_indexes = [output_word2idx['<SOS>']]
    for _ in range(max_output_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]])
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == output_word2idx['<EOS>']:
            break
    trg_tokens = [output_idx2word[idx] for idx in trg_indexes]
    return ''.join(trg_tokens[1:-1])

# 测试翻译
for test_sentence in input_sentences:
    #test_sentence = input_sentences
    translation = translate_sentence(test_sentence, model, input_word2idx, output_idx2word, max_input_len, max_output_len)
    print(f'输入: {test_sentence}')
    print(f'输出: {translation}')
