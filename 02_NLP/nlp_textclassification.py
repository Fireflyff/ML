import time

import torch.optim
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchtext.data import to_map_style_dataset
from torchtext.datasets import AG_NEWS

# 定义模型
from torchtext.vocab import build_vocab_from_iterator

from torchtext.data import get_tokenizer


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_label):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, emb_dim, sparse=True)
        self.FC = nn.Linear(emb_dim, num_label)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.FC.weight.data.uniform_(-initrange, initrange)
        self.FC.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.FC(embedded)


# 定义训练过程
def train(model, dataloader):
    model.train()
    total_acc = 0
    start_time = time.time()
    log_interval = 500
    total_count = 0
    for idx, (texts, labels, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        pred_labels = model(texts, offsets)
        loss = criterion(pred_labels, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_acc += (pred_labels.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        if idx % log_interval == 0 and idx > 0:
            print(f"时间：{time.time() - start_time}，准确率：{total_acc / total_count}")
            total_acc, total_count = 0, 0
            start_time = time.time()
    scheduler.step()


def evaluate(model, dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (texts, labels, offsets) in enumerate(dataloader):
            pred_labels = model(texts, offsets)
            total_acc += (pred_labels.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
        return total_acc / total_count


def predict(text):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        ouput = model(text, torch.tensor([0]))
        return ouput.argmax(1).item() + 1


# 获取词汇的token并进行预处理
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return text_list.to(device), label_list.to(device), offsets.to(device)


if __name__ == '__main__':
    # 参数设置
    batch_size = 64
    epoch = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    train_iter, test_iter = AG_NEWS(root="./dataset/NLP_datasets/AG_NEWS")
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])


    # 定义pipeline
    tokenizer = get_tokenizer('basic_english')

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1
    train_dataloder = DataLoader(split_train_, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    # 模型训练与评估
    vocab_size = len(split_train_)
    emb_dim = 32
    num_label = 4
    ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}
    total_accu = None
    LR = 4.
    # 定义优化器和损失函数
    model = TextClassificationModel(vocab_size, emb_dim, num_label).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    for e in range(1, epoch + 1):
        epoch_start_time = time.time()
        train(model, train_dataloder)
        accu_val = evaluate(model, valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
            pass
        else:
            total_accu = accu_val
        print(f"epoch: {e}, valid accuracy: {accu_val}")

    text = "Haden has been in a good mood recently. He made an excited expression during the training. " \
           "The official website of the 76ers also showed a photo with an article: Who made Beard so excited? " \
           "Later, the official website of the 76ers revealed that he was training with Maxi. As we all know, " \
           "Maxi is a promising new star. Last season, he replaced Simmons, " \
           "which is the main reason why the 76ers were willing to trade the latter. " \
           "Moreover, he and Harden are very cooperative in tactics, " \
           "and once said that he likes playing with Harden. " \
           "During the off-season, Harden voluntarily reduced his salary by 15 million yuan. " \
           "The 76 recruited three players, House, Tucker and Harrell. " \
           "The depth and strength of the team have been greatly improved, " \
           "especially in defense and substitute scoring. This is certainly the reason why Harden is happy. " \
           "He will try his best to attack the championship in the new season."
    print(f"这是一则关于 {ag_news_label[predict(text)]} 的新闻")
