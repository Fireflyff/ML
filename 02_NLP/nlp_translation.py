# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        """添加句子函数，将句子转化为对应的数值序列，输入参数sentence是一条句子"""
        # 对句子进行分割，获取词汇表
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """添加词汇函数，将词汇转化成对应的数值"""
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """字符串规范化函数，参数 s 代表传入的字符串"""
    # 去掉两侧的空白符，再去掉每个单词的重音标志
    s = unicodeToAscii(s.lower().strip())
    # 在.!?前加一个空格
    s = re.sub(r"([.!?])", r" \1", s)
    # 使用正则表达式将字符串中不是大小写字母和正常标点的都替换成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


data_path = "../NLP_datasets/human_name/data/eng-fra.txt"


def readLangs(lang1, lang2):
    """读取语言函数，参数lang1是源语言的名字，参数lang2是目标语言的名字
        返回对应的class Lang对象，以及语言对列表"""
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


# 设置组成句子中单词或标点的最多个数
MAX_LENGTH = 10
# 选择带有指定前缀的语言特征数据作为训练数据
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    """语言对过滤函数，参数p代表输入的语言对"""
    # p[0]代表英语句子，对它进行划分，它的长度应小于最大长度 MAX_LENGTH 并且要以指定的前缀开头
    # p[1]代表法文句子，对它进行划分，它的长度应小于最大长度 MAX_LENGTH
    return len(p[0].split(' ')) < MAX_LENGTH and \
        p[0].startswith(eng_prefixes) and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterpairs(pairs):
    """对多个语言对列表进行过滤，参数pairs代表语言对组成的列表"""
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2):
    """将所有字符串数据向数值型数据的映射以及过滤语言对参数lang1，lang2分别代表源语言和目标语言的名字"""
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    pairs = filterpairs(pairs)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs


def tensorFromSentence(lang, sentence):
    """将文本句子转换为张量，参数lang代表传入的Lang的实例化对象，sentence是预转换的句子"""
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


input_lang, output_lang, pairs = prepareData('eng', 'fra')


def tensorFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    output_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, output_tensor)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """"""
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        """
        :param input: 源语言的Embedding层输入张量
        :param hidden: 编码器层gru的初始隐层张量
        :return:
        """
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        """初始化隐层张量函数"""
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        """初始化函数有两个参数，hidden_size代表解码器中GRU的输入尺寸，也是它的隐层节点数
            output_size代表整个解码器的输出尺寸，也是我们希望得到的指定尺寸即目标语言的词表大小"""
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # 实例化一个nn中的Embedding层对象，它的参数output这里目标语言的此表大小
        # hidden_size表示目标语言的词嵌入维度
        self.embedding = nn.Embedding(output_size, hidden_size)
        # 实例化GRU对象，输入参数都是hidden_size，代表它的输入尺寸和隐层节点数相同
        self.gru = nn.GRU(hidden_size, hidden_size)
        # 实例化线性层，对GRU的输出做线性变化，我们希望的输出尺寸output_size
        # 因此它的两个参数分别是hidden_size,output_size
        self.out = nn.Linear(hidden_size, output_size)
        # 最后使用softmax进行处理，以便分类
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """解码器的前向逻辑函数中，参数有两个，input代表目标语言的Embedding层输入张量
            hidden代表解码器GRU的初始隐层张量"""
        output = self.embedding(input).view(1, 1, -1)
        # 使用relu函数对输出进行处理，根据relu函数的特性，将使Embedding矩阵更稀疏，以防止过拟合
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        """初始化隐层张量函数"""
        # 将隐层张量初始化为1 * 1 * self.hidden_size大小的 0 张量
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        """
        :param hidden_size: 解码器中GRU的输入尺寸，也是它的隐层节点数
        :param output_size: 整个解码器的输出尺寸，也是我们希望得到的指定尺寸即目标语言的词表大小
        :param dropout_p: 使用dropout层时的置零比率，默认0.1
        :param max_lenght: 句子的最大长度
        """
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        # 实例化一个Embedding层
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """forward函数的输入参数有三个，分别是源数据输入张量，初始的隐层张量，以及解码器的输出张量"""
        embedded = self.embedding(input).view(1, 1, -1)
        # 使用dropout进行随机丢弃，防止过拟合
        embedded = self.dropout(embedded)
        # 进行attention的权重计算，将Q、K进行拼接，做一次线性变化，最后使用softmax处理获得结果
        # input等于hidden
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        """初始化隐层张量函数"""
        return torch.zeros(1, 1, self.hidden_size, device = device)


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    """

    :param input_tensor: 源语言输入张量
    :param taget_tensor: 目标语言输入张量
    :param encoder: 编码器实例化对象
    :param decoder: 解码器实例化对象
    :param encoder_optimizer: 编码器优化方法
    :param decoder_optimizer: 解码器优化方法
    :param criterion: 损失函数计算方法
    :param max_length: 句子的最大长度
    :return:
    """
    # 初始化隐层张量
    encoder_hidden = encoder.initHidden()
    # 编码器和解码器优化器梯度归0
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # 根据源文本和目标文本张量获得对应的长度
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # 初始化编码器输出张量，形状是max_length * encoder.hidden_size的 0 张量
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    # 设置损失为0
    loss = 0
    # 循环遍历输入张量索引
    for ei in range(input_length):
        # 根据索引从input_tensor取出对应的单词的张量表示，和初始化隐层张量一同传入encoder对象中
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        # 将每次获得的输出encoder_output（三维张量），使用[0, 0]降两维变成依次存入encoder_outputs
        # 这样encoder_outputs每一行存的都是对应的句子中每个单词通过编码器的输出结果
        encoder_outputs[ei] = encoder_output[0, 0]
    # 初始化解码器的第一个输入，即起始符
    decoder_input = torch.tensor([[SOS_token]], device=device)
    # 初始化解码器的隐层张量即编码器的隐层输出
    decoder_hidden = encoder_hidden
    # 根据随机数与teacher_forcing_ratio对比判断是否使用teacher_forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # 如果使用teacher_forcing:
    if use_teacher_forcing:
        # 循环遍历目标张量索引
        for di in range(target_length):
            # 将decoder_input,decoder_hidden,encoder_outputs即attention中的QKV,
            # 传入解码器对象中，获得decoder_output,decoder_hidden,decoder_attention
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 使用teacher_forcing，即使用正确答案来计算损失
            loss += criterion(decoder_output, target_tensor[di])
            # 并强制将下一次的编码器输入设置为'正确答案'
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            # 将decoder_input,decoder_hidden,encoder_outputs即attention中的QKV,
            # 传入解码器对象中，获得decoder_output,decoder_hidden,decoder_attention
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # 使用teacher_forcing，即使用正确答案来计算损失
            loss += criterion(decoder_output, target_tensor[di])
            topv, topi = decoder_output.topk(1)
            if topi.squeeze().item() == EOS_token:
                break
            # detach使得这个decoder_input与模型构建的张量图无关
            decoder_input = topi.squeeze().detach()

    # 误差进行反向传播
    loss.backward()
    # 编码器和解码器进行优化即参数更新
    encoder_optimizer.step()
    decoder_optimizer.step()
    # 最后返回平均损失
    return loss.item() / target_length


def timeSince(since):
    """获得每次打印的训练耗时, since是训练开始时间"""
    # 获取当前时间
    now = time.time()
    # 获得时间差
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    # 使用预定义的SGD作为优化器，将参数和学习率传入其中
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # 选择损失函数
    criterion = nn.NLLLoss()
    # 根据设置迭代进行循环
    for iter in range(1, n_iters + 1):
        # 每次从语言对列表中随机取出一条作为训练语句
        training_pair = tensorFromPair(random.choice(pairs))
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        # 通过train获得模型的损失
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss
        # 打印日志
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter/n_iters * 100, print_loss_avg))
        # 绘制间隔
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # 绘制损失曲线
    plt.figure()
    plt.plot(plot_losses)
    plt.show()


hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=.1,).to(device)
n_iters = 75000
print_every = 5000

trainIters(encoder1, attn_decoder1, n_iters, print_every)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    # 评估阶段不计算梯度
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoder_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoder_words.append('<EOS>')
                break
            else:
                decoder_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        return decoder_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=5):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print(' ')

evaluateRandomly(encoder1, attn_decoder1)

sentence = "we re both teachers ."
output_words, attentions = evaluate(encoder1, attn_decoder1, sentence)
print(output_words)
plt.matshow(attentions.numpy())
plt.show()