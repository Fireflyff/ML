from datasets import load_from_disk
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW


# 定义数据集
class Dataset(Dataset):
    def __init__(self, split):
        dataset = load_from_disk("../../NLP_datasets/ChnSentiCorp/" + split)

        def f(data):
            return len(data["text"]) > 40
        self.dataset = dataset.filter(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        # 切分一句话为前半句和后半句
        sentence1 = text[:20]
        sentence2 = text[20:40]
        label = 0
        # 有一半的概率把后半句替换为一句无关的话
        if random.randint(0, 1) == 0:
            j = random.randint(0, len(self.dataset) - 1)
            sentence2 = self.dataset[j]["text"][20:40]
            label = 1
        return sentence1, sentence2, label


dataset = Dataset("train")
token = BertTokenizer.from_pretrained("bert-base-chinese")


def collate_fn(data):
    sents = [i[:2] for i in data]
    labels = [i[2] for i in data]
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=45,
        padding="max_length",
        return_tensors='pt',
        return_length=True,
        add_special_tokens=True)
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, labels


loader = DataLoader(dataset=dataset,
                    batch_size=8,
                    collate_fn=collate_fn,
                    shuffle=True,
                    drop_last=True)


pretrained = BertModel.from_pretrained("bert-base-chinese")
for param in pretrained.parameters():
    param.requires_grad_(False)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        # out = self.fc(out[0][:, 0])
        out = self.fc(out[1])
        # todo out[0][0] != out[1] ==> ???
        # todo:answer：out[0]：last_hidden_state，out[1]：pooler_output
        # todo:out[1]为out[0][0]（[CLS]）经过dense和activation之后的结果
        # todo:两者均可训练
        out = out.softmax(dim=1)
        return out


model = Model()
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
    if i == 300:
        break


def model_test():
    model.eval()
    correct = 0
    total = 0
    loader_test = DataLoader(dataset=Dataset("test"),
                             batch_size=32,
                             collate_fn=collate_fn,
                             shuffle=True,
                             drop_last=True
                             )
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        if i == 5:
            break
        print(i)
        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += len(labels)
    print(correct / total)

model_test()
