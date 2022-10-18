import torch
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator

train_iter = AG_NEWS(split='train')
print(len(train_iter))
tokenizer = get_tokenizer('basic_english')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["ii", "<unk>", "YY"])
vocab.set_default_index(vocab["<unk>"])
print("YY", len(vocab))
print(vocab.lookup_tokens([i for i in range(15)]))
print(dict((i, vocab.lookup_token(i)) for i in range(15)))

sentence = "YY is YYDS! YY ht qwertyuiiop, 78"
print(vocab(tokenizer(sentence)))

# 定义pipeline
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

print(text_pipeline("ying is YYDS!"))
print(label_pipeline(0))


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


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


dataloader = DataLoader(train_iter, batch_size=12, shuffle=False, collate_fn=collate_batch)
for i in dataloader:
    print(i)









# train_dataset.get_vocab()
# x = torch.tensor([3,4,5])
# print(x[0].item())
# print(x)
# a1 = torch.tensor([1,2,3,4,5,6,7])
# a2 = torch.tensor([2,3,4,5,6,7,8])
# torch.add(a1, a2, out=x)
# print(x, x.size())
# import torch.nn as nn
# embedding = nn.Embedding(6, 2, 3)
# nn.EmbeddingBag()
# print(embedding.weight)
# **********

# A = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[2, 3, 4], [5, 6, 7], [8, 9, 10]]])
# print(A, A.size())
# print(A.permute(1, 0, 2))
# print(A.permute(2, 1, 0))
# print(A.permute(2, 0, 1))
# **********
# num = 5
# embedding.weight.data.uniform_(-num, num)
# print(embedding.weight.size())
# print("size: ", embedding.weight.size(1), embedding.weight.size(0))
# print(embedding.weight)

# print(torch.mean(embedding.weight), torch.var(embedding.weight))
# t = torch.ones((100, 100)).to(int)
# # print(t)
# # print(embedding(t))
# print("#"*19)
# print(embedding(torch.tensor([[2,1],[1,2]])))
#
# label = torch.Tensor(5,1,)
# print(label)
# print(label.fill_(3))

# L = nn.Linear(5, 3)
# print(L)
# print(L.weight)
# print(L.weight.size())
# print(L.bias)
# torch.manual_seed(0)
# vocab_size = 4
# embedding_dim = 3
# weight = torch.randn(4, 3)
#
# linear_layer = nn.Linear(4, 3, bias=False)
# linear_layer.weight.data = weight.T
# emb_layer = nn.Embedding(4, 3)
# emb_layer.weight.data = weight
#
# idx = torch.tensor(2)
# word = torch.tensor([0, 0, 1, 0]).to(torch.float)
# print(emb_layer(idx))
# print(linear_layer(word))
