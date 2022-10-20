import glob
# 用于获取常见字母及字符规范化
import string
import unicodedata
import random
import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

all_letters = string.ascii_letters + " .,;'"
# 获取常用字符数量
n_letters = len(all_letters)


# 去掉一些语言中的重音标记
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


data_path = "../NLP_datasets/human_name/data/names/"


def readLines(filename):
    """从文件中读取每一行加载在内存中形成列表"""
    # 打开指定文件并读取所有内容，使用strip()去除两侧空白符，然后以'\n'进行切分
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    # 对应每一个lines列表中的名字进行Ascii转换，使其规范化，最后返回一个名字列表
    return [unicodeToAscii(line) for line in lines]


category_lines = {}
all_category = []
for filename in glob.glob(data_path + '*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_category.append(category)
    lines = readLines(filename)
    category_lines[category] = lines


def lineToTensor(line):
    """将所有人名转化为onehot张量表示"""
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        """初始化函数中有4个参数, 分别代表RNN输入最后一维尺寸, RNN的隐层最后一维尺寸, RNN层数"""
        super(RNN, self).__init__()
        # 将hidden_size与num_layers传入其中
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 实例化预定义的nn.RNN, 它的三个参数分别是input_size, hidden_size, num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        # 实例化nn.Linear, 这个线性层用于将nn.RNN的输出维度转化为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定的Softmax层, 用于从输出层获得类别结果
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        """完成传统RNN中的主要逻辑, 输入参数input代表输入张量, 它的形状是1 x n_letters
           hidden代表RNN的隐层张量, 它的形状是self.num_layers x 1 x self.hidden_size"""
        # 因为预定义的nn.RNN要求输入维度一定是三维张量, 因此在这里使用unsqueeze(0)扩展一个维度
        input = input.unsqueeze(0)
        # 将input和hidden输入到传统RNN的实例化对象中，如果num_layers=1, rr恒等于hn
        rr, hn = self.rnn(input, hidden)
        # 将从RNN中获得的结果通过线性变换和softmax返回，同时返回hn作为后续RNN的输入
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        """初始化隐层张量"""
        # 初始化一个（self.num_layers, 1, self.hidden_size）形状的0张量
        return torch.zeros(self.num_layers, 1, self.hidden_size)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """初始化函数的参数与传统RNN相同"""
        super(LSTM, self).__init__()
        # 将hidden_size与num_layers传入其中
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 实例化预定义的nn.LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, c):
        """在主要逻辑函数中多出一个参数c, 也就是LSTM中的细胞状态张量"""
        # 使用unsqueeze(0)扩展一个维度
        input = input.unsqueeze(0)
        rr, (hn, c) = self.lstm(input, (hidden, c))
        return self.softmax(self.linear(rr)), hn, c

    def initHidden(self):
        """初始化函数不仅初始化hidden还要初始化细胞状态c, 它们形状相同"""
        c = hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, c


# GRU与传统RNN的外部形式相同, 都是只传递隐层张量, 因此只需要更改预定义层的名字
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 实例化预定义的nn.GRU, 它的三个参数分别是input_size, hidden_size, num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        rr, hn = self.gru(input, hidden)
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# 因为是onehot编码，输入张量最后一维的尺寸就是n_letters
input_size = n_letters
# 定义隐层的最后一维尺寸大小
n_hidden = 128
# 输出尺寸为语言类别总数n_categories
n_categories = len(all_category)
output_size = n_categories
# num_layer使用默认值
num_layer = 1

# 调用
rnn = RNN(n_letters, n_hidden, n_categories, num_layer)
lstm = LSTM(n_letters, n_hidden, n_categories, num_layer)
gru = GRU(n_letters, n_hidden, n_categories, num_layer)

# rnn_output, next_hidden = rnn(input, hidden)
# print("rnn:", rnn_output)
# lstm_output, next_hidden, c = lstm(input, hidden, c)
# print("lstm:", lstm_output)
# gru_output, next_hidden = gru(input, hidden)
# print("gru:", gru_output)


def categoryFromOutput(output):
    """从输出结果中获取指定类别，参数为输出张量output"""
    # 从输出张量中返回最大的值和索引对象
    top_n, top_i = output.topk(1)
    # top_i对象中获取索引的值
    category_i = top_i[0].item()
    # 根据索引值获得对应语言类别，返回语言类别和索引值
    return all_category[category_i], category_i


def randomTrainingExample():
    """随机产生训练数据"""
    # 首先使用random的choice方法从all_categories随机选择一个类别
    category = random.choice(all_category)
    # 然后在通过category_lines字典取category类别对应的名字列表
    # 之后再从列表中随机取一个名字
    line = random.choice(category_lines[category])
    # 接着将这个类别在所有类别列表中的索引封装成tensor, 得到类别张量category_tensor
    category_tensor = torch.tensor([all_category.index(category)], dtype=torch.long)
    # 最后, 将随机取到的名字通过函数lineToTensor转化为onehot张量表示
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


# 定义损失函数为nn.NLLLoss，因为RNN的最后一层是nn.LogSoftmax, 两者的内部计算逻辑正好能够吻合.
criterion = nn.NLLLoss()
# 设置学习率
learning_rate = 0.015


def unit_train(category_tensor, line_tensor, model):
    """定义训练函数, 它的两个参数是category_tensor类别的张量表示, 相当于训练数据的标签,
        line_tensor名字的张量表示, 相当于对应训练数据"""
    # 在函数中, 首先通过实例化对象model初始化隐层张量, lstm返回hidden和c
    Hidden = model.initHidden()
    if isinstance(Hidden, tuple):
        tmp = (torch.tensor([0]), *Hidden)
    else:
        tmp = (torch.tensor([0]), Hidden)

    # 然后将模型结构中的梯度归0
    model.zero_grad()
    # 下面开始进行训练, 将训练数据line_tensor的每个字符逐个传入rnn之中, 得到最终结果
    for i in range(line_tensor.size()[0]):
        # rnn或者gru:返回 output 和 hidden
        # lstm：返回output、hidden 以及 c
        tmp = model(line_tensor[i], *tmp[1:])
    output = tmp[0]
    # 因为我们的rnn对象由nn.RNN实例化得到, 最终输出形状是三维张量, 为了满足于category_tensor
    # 进行对比计算损失, 需要减少第一个维度, 这里使用squeeze()方法
    loss = criterion(output.squeeze(0), category_tensor)

    # 损失进行反向传播
    loss.backward()
    # 更新模型中所有的参数
    for p in model.parameters():
        # 将参数的张量表示与参数的梯度乘以学习率的结果相加以此来更新参数
        p.data.add_(-learning_rate, p.grad.data)
    # 返回结果和损失的值
    return output, loss.item()


def timeSince(since):
    """获得每次打印的训练耗时, since是训练开始时间"""
    # 获取当前时间
    now = time.time()
    # 获得时间差
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# 设置训练的迭代次数
n_iters = 10000
# 设置结果的打印间隔
print_every = 500
# 设置绘制损失曲线上的绘制间隔
plot_every = 30


def train(train_type_fn):
    # 每个制图间隔损失保存列表
    all_losses = []
    # 获得训练开始时间戳
    start = time.time()
    # 设置初始间隔损失为0
    current_loss = 0
    # 从1开始进行训练迭代，共n_iters次
    for iter in range(1, n_iters + 1):
        # 获取训练数据和对应的类别
        category, line, category_tensor, line_tensor = randomTrainingExample()
        # 将训练数据和对应的类别张量传入unit_train
        output, loss = unit_train(category_tensor, line_tensor, train_type_fn)
        # 计算制图间隔中的损失
        current_loss += loss
        # 每迭代间隔次打印loss信息
        if iter % print_every == 0:
            # 取该迭代步上的output通过categoryFromOutput函数获得对应的类别和类别索引
            guess, guess_i = categoryFromOutput(output)
            # 然后和真实的类别category做比较，如果相同则打对号，否则打叉号
            correct = '✓' if guess == category else '✗ (%s)' % category
            # 打印迭代步, 迭代步百分比, 当前训练耗时, 损失, 该步预测的名字, 以及是否正确
            print('%d %d%% (%s) %.4f %s / %s %s' % (
            iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # 每绘制间隔次打印绘图信息
        if iter % plot_every == 0:
            #将保存间隔中的平均损失到all_losses列表中
            all_losses.append(current_loss / plot_every)
            # 间隔损失重置为0
            current_loss = 0

    # 返回对应的总损失列表和训练耗时
    return all_losses, int(time.time() - start)

# 开始训练传统的RNN LSTM 以及 GRU 模型并制作对比图
# 返回各自的全部损失，以及训练耗时用于制图
all_losses1, period1 = train(rnn)
all_losses2, period2 = train(lstm)
all_losses3, period3 = train(gru)

# # 绘制损失对比曲线，训练耗时对比柱状图
# # 创建画布0
# plt.figure(0)
# # 绘制损失对比曲线
# plt.plot(all_losses1, label='RNN')
# plt.plot(all_losses2, color='red', label='LSTM')
# plt.plot(all_losses3, color='orange', label='GRU')
# plt.legend(loc='upper left')
# plt.show()
# # 创建画布1
# plt.figure(1)
# x_data = ['RNN', 'LSTM', 'GRU']
# y_data = [period1, period2, period3]
# # 耗时训练时对比柱状图
# plt.bar(range(len(x_data)), y_data, tick_label=x_data)
#
# plt.show()


def evaluate(line_tensor, model):
    """评估函数, 和训练函数逻辑相同, 参数是 line_tensor 代表名字的张量表示"""
    # 初始化隐层张量
    Hidden = model.initHidden()
    if isinstance(Hidden, tuple):
        tmp = (torch.tensor([0]), *Hidden)
    else:
        tmp = (torch.tensor([0]), Hidden)

    # 将评估数据 line_tensor 的每个字符逐个传入 model 之中
    for i in range(line_tensor.size()[0]):
        tmp = model(line_tensor[i], *tmp[1:])
    # 获得输出结果
    output = tmp[0]
    return output.squeeze(0)


def predict(input_line, model, n_predictions=3):
    """预测函数, 输入参数input_line代表输入的名字,
           n_predictions代表需要取最有可能的top个"""
    # 首先打印输入
    print('\n> %s' % input_line)
    # 以下操作的相关张量不进行求梯度
    with torch.no_grad():
        # 使输入的名字转化成张量表示，并使用evaluate函数获得预测输出
        output = evaluate(lineToTensor(input_line), model)
        # 从预测的输出中取前3个最大的值及其索引
        topv, topi = output.topk(n_predictions, 1, True)
        # 创建盛装结果的列表
        predictions = []
        # 遍历 n_predictions
        for i in range(n_predictions):
            # 从 topv 中取出的 output 值
            value = topv[0][i].item()
            # 取出索引并找到对应的类别
            category_index = topi[0][i].item()
            # 打印 output 的值，和对应的类别
            print('(%.2f) %s' % (value, all_category[category_index]))
            # 将结果装进 predictions 中
            predictions.append([value, all_category[category_index]])


for model in [rnn, lstm, gru]:
    print("-"*18)
    predict('Dovesky', model)
    predict('Jackson', model)
    predict('Satoshi', model)