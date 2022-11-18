import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments


def getDataLists(dataframe):
    if 'target' in dataframe.columns:
        return list(dataframe.text), list(dataframe.target)
    else:
        return list(dataframe.text)


data_path = '../../../NLP_datasets/disaster_tweets_classification'
dfTrain = pd.read_csv(f"{data_path}/train.csv")
dfTest = pd.read_csv(f"{data_path}/test.csv")

trainText, trainLabels = getDataLists(dfTrain)
testText = getDataLists(dfTest)
trainTexts, validationTexts, trainLabels, validationLabels = train_test_split(trainText, trainLabels,
                                                                              test_size=.2, random_state=42,
                                                                              shuffle=True)
modelCheckpoint = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(modelCheckpoint)

trainEncodings = tokenizer(trainTexts, truncation=True, padding=True)

validationEncodings = tokenizer(validationTexts, truncation=True, padding=True)
testEncodings = tokenizer(testText, truncation=True, padding=True)


class Data(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


trainDataset = Data(trainEncodings, trainLabels)
validationDataset = Data(validationEncodings, validationLabels)
per_device_train_batch_size = 16
per_device_eval_batch_size = 64

trainingArgs = TrainingArguments(
    output_dir='./results',
    logging_dir='./logs',
    logging_steps=10,
    num_train_epochs=3,
    warmup_steps=500,
    learning_rate=4e-05,
    weight_decay=0.02,
    # report_to="none"
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,
    args=trainingArgs,
    train_dataset=trainDataset,
    eval_dataset=validationDataset
)

trainer.train()
predictions = trainer.predict(validationDataset)
yPreds = np.argmax(predictions.predictions, axis=1)
print(classification_report(yPreds, validationDataset.labels))

dfTest['target_dummy'] = 0
testDataset = Data(testEncodings, list(dfTest['target_dummy']))
predictionsTest = trainer.predict(testDataset)

yPredsTest = np.argmax(predictionsTest.predictions, axis=1)
dfSubmission = pd.DataFrame()
dfSubmission['id'] = dfTest['id']
dfSubmission['target'] = yPredsTest

dfSubmission.to_csv('submission.csv', index=False)
