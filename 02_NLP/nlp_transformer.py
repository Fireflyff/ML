import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import copy
import matplotlib.pyplot as plt
from pyitcast.transformer_utils import Batch, get_std_opt, LabelSmoothing, SimpleLossCompute,\
    run_epoch, greedy_decode


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # d_model:词嵌入维度
        # vocab: 词表的大小
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # max_len：每个句子的最大长度
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        # 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # 偶数维度
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # register_buffer通常用于注册不应该被视为模型参数的缓冲区
        # 缓冲区可以使用给定的名称作为属性访问
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 输出最终的编码
        x += Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


"""
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(8, 0)
y = pe(Variable(torch.zeros(1, 100, 8)))
# 词向量中某个元素随该词在句子中位置的不同而变化的曲线
# plt.plot(np.arange(100), y[0, :, 3:6].data.numpy().T)
# 每个词向量的pos_embedding
plt.plot(np.arange(8), y[0, 3:6, :].data.numpy().T)
plt.legend(["dim %d" % p for p in [3, 4, 5]])
plt.show()
"""


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.tril(np.ones(attn_shape), k=0).astype('uint8')
    # 0：进行遮挡
    # 1：可以看到
    return torch.from_numpy(subsequent_mask)


def attention(query, key, value, mask=None, dropout=None):
    # query的size：(句子的数量，head的大小，句子的长度，每个head的embedding_size)
    """
    ：关于mask的mark：表示一个句子中哪些单词需要忽略。
    ：Encoder的Multi-Head Attention为input Embedding的mask
    ：Decoder的Masked Multi-Head Attention 为output Embedding的mask
    ：Decoder的Multi-Head Attention为Encoder输出结果的mask,一般为input Embedding的mask
    """
    d_k = query.size(-1)
    # scores的size：(句子的数量，head的大小，句子中对应词汇的相似度)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print("scores的size： ", scores.size())
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    # 进行N次深拷贝，使每个module成为独立的层
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        # embedding_dim：词嵌入的维度
        # head：头数
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % head == 0
        # 每个注意力机制获得的分割词向量的维度
        self.d_k = embedding_dim // head
        self.head = head
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 每个head去关注词向量的部分信息，最终结合所有head的结果->
        # (batch, head, sentence_length, d_k)->(batch, sentence_length, head*d_k)->
        # (batch, sentence_length->embedding_size),即原数据x的维度
        if mask is not None:
            mask = mask.unsqueeze(0)
        # 样本个数
        batch_size = query.size(0)
        # zip(a,b) a,b长度可以不一致，以短的那个为主
        # (batch_size,head,num_words,embedding_size) →
        # (batch_size,head,num_words,d_k)
        # embedding_size = head*d_k
        # Q,K,V的维度均为 (batch_size,head,num_words,d_k)
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value))]
        # x, self.attn 的维度与QKV一致: (batch_size,head,num_words,d_k)
        # print("多头注意力机制中的forward：", query.size(), mask.size())
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # print("多头注意力机制的embedding以及相关系数：", x.size(), self.attn.size())
        # contiguous类似深拷贝
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        return self.linears[-1](x)


# 通过PositionwiseFeedForward来实现前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        前馈全连接层
        d_model: 词嵌入维度
        d_ff: 隐藏层维度
        dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


# 通过LayerNorm实现规范化层的类
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """初始化函数有两个参数, 一个是features, 表示词嵌入的维度,
            另一个是eps它是一个足够小的数, 在规范化公式的分母中出现,
            防止分母为0.默认是1e-6."""
        super(LayerNorm, self).__init__()
        # scale
        self.a2 = nn.Parameter(torch.ones(features))
        # shift
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


# 使用SublayerConnection来实现子层连接结构的类
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        # size：词嵌入的维度大小
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        # todo return x + self.dropout(self.norm(sublayer(x)))
        return x + self.dropout(self.norm(sublayer(x)))
        # return x + self.dropout(sublayer(self.norm(x)))


# 使用EncoderLayer类实现编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        :param size: 词嵌入维度，即编码器层的大小
        :param self_attn: 多头自注意力子层实例化对象
        :param feed_forward:前馈全连接层实例化对象
        :param dropout:
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# 实现编码器
class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        :param layer: 编码器层
        :param N: 编码器层的个数
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# 使用DecoderLayer的类实现解码器层
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        :param size: 词嵌入维度
        :param self_attn: 多头注意力对象，Q=K=V
        :param src_attn: 多头注意力对象，Q!=K=V
        :param feed_forward: 前馈全连接对象
        :param dropout:
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        # 编码器的语义存储变量memory
        m = memory
        # target_mask：未来的信息被遮挡
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # source_mask：遮蔽对结果没有意义的字符而产生的注意力值
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        return self.sublayer[2](x, self.feed_forward)


# 实现解码器
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


# 生成器(将线性层和softmax计算层一起实现)
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)


# 使用EncoderDecoder类来实现编码器-解码器结构
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        """
        :param encoder: 编码器对象
        :param decoder: 解码器对象
        :param source_embed: 源数据嵌入函数
        :param target_embed: 目标数据嵌入函数
        :param generator: 类别生成器对象
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=.1):
    """
    :param source_vocab: 源数据词汇总数
    :param target_vocab: 目标数据词汇总数
    :param N: 编码器和解码器堆叠数
    :param d_model: 词向量维度
    :param d_ff: 前馈全连接网络中变换矩阵的维度
    :param head: 多头注意力结构的头数
    :param dropout:
    :return:
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
                           Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                           nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
                           nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
                           Generator(d_model, target_vocab)
                           )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def data_generator(V, batch, num_batch):
    """
    :param V: 随机生成数字的最大值+1
    :param batch: 每次输送给模型更新一次参数的数据量
    :param num_batch: 一次输送num_batch次完成一轮
    :return:
    """
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        # 解码器解码，会使用起始标志列作为输入，即 1
        data[:, 0] = 7777
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)
        # 使用Batch对source和target进行对应批次的掩码张量生成, 最后使用yield返回
        yield Batch(source, target)


def run(model, loss, epochs=10):
    for epoch in range(epochs):
        model.train()
        run_epoch(data_generator(V, 8, 20), model, loss)
        model.eval()
        run_epoch(data_generator(V, 8, 5), model, loss)
    # 模型进入测试模式
    model.eval()
    source = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    # 在这个工具包中，0表示遮挡
    source_mask = Variable(torch.ones(1, 1, 10))
    result = greedy_decode(model, source, source_mask, max_len=20, start_symbol=7777)
    print(result)


if __name__ == '__main__':
    V = 7778
    batch = 1
    num_batch = 1
    model = make_model(V, V, N=2)
    # 使用get_std_opt获得模型优化器
    # 该标准优化器基于Adam优化器, 使其对序列到序列的任务更有效.
    model_optimizer = get_std_opt(model)
    # 导入标签平滑工具包, 该工具用于标签平滑, 标签平滑的作用就是小幅度的改变原有标签值的值域
    # 因为在理论上即使是人工的标注数据也可能并非完全正确, 会受到一些外界因素的影响而产生一些微小的偏差
    # 因此使用标签平滑来弥补这种偏差, 减少模型对某一条规律的绝对认知, 以防止过拟合.
    # size：数据的词汇总数
    # smoothing：标签的平滑程度，假说原来的标签的表示值为1，则平滑后它的值域变为【1-smoothing, 1+smoothing】
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    # SimpleLossCompute能够使用标签平滑后的结果进行损失的计算,
    # 损失的计算方法可以认为是交叉熵损失函数.
    loss = SimpleLossCompute(model.generator, criterion, model_optimizer)
    run(model, loss, num_batch)




"""
    # 词嵌入维度是512维
    d_model = 512
    
    # 词表大小是1000
    vocab = 1000
    
    # 置0比率为0.1
    dropout = 0.1
    
    # 句子最大长度
    max_len = 60
    
    # 输入x是一个使用Variable封装的长整型张量, 形状是2 x 4
    x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
    
    
    emb = Embeddings(d_model, vocab)
    embr = emb(x)
    x = embr
    print("原句子为2*4，embedding之后的维度大小：", x.size())
    
    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(x)
    print("加入pos embedding之后的input: ", pe_result)
    
    # 头数head
    head = 8
    
    # 词嵌入维度embedding_dim
    embedding_dim = 512
    
    # 置零比率dropout
    dropout = 0.2
    
    
    # 假设输入的Q，K，V仍然相等
    query = value = key = pe_result
    print("Q、K、V的大小：", query.size())
    
    # 输入的掩码张量mask
    mask = Variable(torch.zeros(8, 4, 4))
    ff = PositionwiseFeedForward(d_model, 64, dropout)
    
    self_attn = MultiHeadedAttention(head, embedding_dim, dropout)
    el = EncoderLayer(embedding_dim, self_attn, ff, dropout)
    el_result = el(x, mask)
"""
