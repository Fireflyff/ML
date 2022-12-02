import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.optim import lr_scheduler
from torch import nn
from torch.optim import Adam
from tqdm.notebook import tqdm

df = pd.read_csv("../../../NLP_datasets/feedback_prize_english/train.csv")


def process_text(df):
    df = df.copy()
    df['full_text'] = df['full_text'].apply(lambda x: x.replace('\n', ' '))
    df['full_text'] = df['full_text'].apply(lambda x: x.strip())
    df['full_text'] = df['full_text'].apply(lambda x: x.lower())
    return df


df = process_text(df)

tokenizer = BertTokenizer.from_pretrained('../input/bertlargecased/bert-large-cased')


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = df[["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]].reset_index()
        self.texts = df[["full_text"]].reset_index()

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        # print(self.labels.loc[idx])
        return np.array(self.labels.loc[idx].values[1:]).astype(float)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return tokenizer(self.texts.loc[idx].values[1],
                         padding='max_length', max_length=512, truncation=True,
                         return_tensors="pt")

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class FeedbackModel(nn.Module):

    def __init__(self, dropout=0.5):
        super(FeedbackModel, self).__init__()

        self.bert = BertModel.from_pretrained('../input/bertlargecased/bert-large-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 6)

    def forward(self, input_id, mask):
        _, x = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.relu(x)
        final_layer = self.out(x)
        return final_layer


def train(model, train_data, val_data, epochs):

    train, val = Dataset(train_data), Dataset(val_data)
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-5)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=500,
                                                                 eta_min=1e-6)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()
    for epoch_num in range(epochs):

        total_loss_train = 0

        for train_input, train_labels in tqdm(train_dataloader):
            train_labels = train_labels.to(device).float()
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_labels)
            total_loss_train += batch_loss.item()

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()

        total_loss_val = 0
        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

        print(f'Epoch: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Val Loss: '
              f'{total_loss_val / len(val_data): .3f}')


def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    criterion = nn.MSELoss()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_loss_test = 0
    with torch.no_grad():

        for test_input, test_labels in tqdm(test_dataloader):
            test_labels = test_labels.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            loss = criterion(output, test_labels)
            total_loss_test += loss

    print(f'Test Loss: {total_loss_test / len(test_data): .3f}')


np.random.seed(42)
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                     [int(.9 * len(df)), int(.95 * len(df))])

print(len(df_train), len(df_val), len(df_test))

EPOCHS = 6
model = FeedbackModel()

train(model, df_train, df_val, EPOCHS)
evaluate(model, df_test)
torch.save(model.cpu().state_dict(), "BERT.bin")

test = pd.read_csv("../input/feedback-prize-english-language-learning/test.csv")
test.head()


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.texts = df[["full_text"]]

    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return tokenizer(self.texts.loc[idx].values[0],
                         padding='max_length', max_length=512, truncation=True,
                         return_tensors="pt")

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        return batch_texts


def predict(model, test_data):
    prediction = []
    test = TestDataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    with torch.no_grad():
        for test_input in test_dataloader:

            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            for pred in output.cpu():
                prediction.append(np.array([min(max(1.0, i), 5.0) for i in np.array(pred)]))
    return np.array(prediction)


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
model.eval()


prediction = predict(model, test)
submission = pd.read_csv("../input/feedback-prize-english-language-learning/sample_submission.csv")
submission.cohesion = prediction[:, 0]
submission.syntax = prediction[:, 1]
submission.vocabulary = prediction[:, 2]
submission.phraseology = prediction[:, 3]
submission.grammar = prediction[:, 4]
submission.conventions = prediction[:, 5]


submission.to_csv("submission_4.csv", index=False)
