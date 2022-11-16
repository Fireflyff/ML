import datasets
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW


# 定义数据集
class Dataset(Dataset):
    def __init__(self, split):
        # split = "train"
        # Dataset({
        #     features: ['text', 'label'],
        #     num_rows: 9600
        # })
        self.dataset = datasets.load_from_disk("../../NLP_datasets/ChnSentiCorp/" + split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        label = self.dataset[item]["label"]
        return text, label


dataset = Dataset("train")
token = BertTokenizer.from_pretrained("bert-base-chinese")


def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding="max_length",
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)
    input_ids = data["input_ids"]  # （batch_size, sqe_length）
    attention_mask = data["attention_mask"]  # （batch_size, sqe_length）
    token_type_ids = data["token_type_ids"]  # （batch_size, sqe_length）
    labels = torch.LongTensor(labels)  # （batch_size,）
    return input_ids, attention_mask, token_type_ids, labels


# 数据加载器
loader = DataLoader(dataset=dataset, batch_size=16, collate_fn=collate_fn, shuffle=True, drop_last=True)
# 加载预训练模型
pretrained = BertModel.from_pretrained("bert-base-chinese")
# 不训练，不需要计算梯度
for param in pretrained.parameters():
    param.requires_grad_(False)


# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # out ==> (last_hidden_state, pooler_output) ==>
        # ((batch_size, sqe_length, hidden_size), (batch_size, hidden_size))
        out = self.fc(out[0][:, 0])
        out = out.softmax(dim=1)
        return out


model = Model()

# 训练
optimizer = AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()
model.train()
for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
    out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if i % 5 == 0:
        out = out.argmax(dim=1)
        accuracy = (out == labels).sum().item() / len(labels)
        print(i, loss.item(), accuracy)
    if i == 50:
        break


# 测试
def model_test():
    model.eval()
    correct = 0
    total = 0
    loader_test = DataLoader(dataset=Dataset("validation"),
                             batch_size=32,
                             collate_fn=collate_fn,
                             shuffle=True,
                             drop_last=True)
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        if i == 5:
            break
        print(i)
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
            out = out.argmax(dim=1)
            correct += (out == labels).sum().item()
            total += len(labels)
    print(correct / total)

model_test()

