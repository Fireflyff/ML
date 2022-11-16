import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW


# 定义数据集
class Dataset(Dataset):
    def __init__(self, split):
        dataset = load_from_disk("../../NLP_datasets/ChnSentiCorp/" + split)

        def f(data):
            return len(data["text"]) > 30
        self.dataset = dataset.filter(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        return text


dataset = Dataset("train")
token = BertTokenizer.from_pretrained("bert-base-chinese")


def collate_fn(data):
    # 编码
    data = token.batch_encode_plus(batch_text_or_text_pairs=data,
                                   truncation=True,
                                   padding="max_length",
                                   max_length=30,
                                   return_tensors='pt',
                                   return_length=True
                                   )
    # 编码之后的数字索引
    input_ids = data["input_ids"]  # (batch_size, sqe_length)
    # 补零的位置是0，其他位置是1
    attention_mask = data["attention_mask"]  # (batch_size, sqe_length)
    # 句子所在的位置
    token_type_ids = data["token_type_ids"]  # (batch_size, sqe_length)
    # 把第15个词固定替换为mask
    labels = input_ids[:, 15].reshape(-1).clone()  # (batch_size,)
    input_ids[:, 15] = token.get_vocab()[token.mask_token]
    # todo: print(token.mask_token) ==> [MASK]
    # token.get_vocab()[token.mask_token] ==> 103
    return input_ids, attention_mask, token_type_ids, labels


loader = DataLoader(dataset=dataset,
                    batch_size=16,
                    collate_fn=collate_fn,
                    shuffle=True,
                    drop_last=True)

pretrained = BertModel.from_pretrained("bert-base-chinese")
# 不训练，不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)


# 定义下游任务
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.decoder = torch.nn.Linear(768, token.vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(token.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        out = self.decoder(out[0][:, 15])
        return out


model = Model()
optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()
model.train()
for epoch in range(5):
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 50 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            print(epoch, i, loss.item(), accuracy)


def model_test():
    model.eval()
    correct = 0
    total = 0
    loader_test = DataLoader(dataset=Dataset("test"),
                             batch_size=32,
                             collate_fn=collate_fn,
                             shuffle=True,
                             drop_last=True)
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        if i == 15:
            break
        print(i)
        with torch.no_grad():
            # torch.Size([32, 21128])
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        # torch.Size([32])
        # tensor([6858, 4638, 3300, 1762, 3300, 3315,  119, 4638, 5632, 1762,  671, 1377,
        #         3614, 3198, 2523, 6814,  679, 2697, 5143, 3221,  679, 6983, 1400, 3322,
        #         4638,  809, 1057, 4638, 4685, 1962, 5125, 1920])
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)
        print(token.decode(input_ids[0]))
        print(token.decode(out[0].reshape(-1)), token.decode(labels[0].reshape(-1)))

    print(correct / total)

model_test()
