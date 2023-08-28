import numpy as np
from tqdm.notebook import tqdm
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Let's import the public training set and take a look
import pandas as pd

df_valid = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/train.csv')
df_valid.head()

model_path = '/kaggle/input/transformers/t5-large'
model = T5ForConditionalGeneration.from_pretrained(model_path).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ################################################ 预训练模型计算 loss ###############################################
valid_score = 0
model.eval()
for index in tqdm(range(df_valid.shape[0])):
    columns = df_valid.iloc[index].values
    scores = []
    input_ids = tokenizer(columns[1]+" <extra_id_0>", return_tensors="pt").input_ids.cuda()
    labels = tokenizer(["<extra_id_0> "+columns[2+p] for p in range(5)], return_tensors="pt", padding=True).input_ids
    minlen = np.min([len(l) for l in labels])
    for p in range(5):
        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=labels[p][:minlen].unsqueeze(0).cuda()).loss.detach().cpu().numpy()
        scores.append(float(loss))
    predict = np.array(list("ABCDE"))[np.argsort(scores)][:3].tolist()
    if columns[7] in predict:
        valid_score += [1, 0.5, 0.333333333333][predict.index(columns[7])]
valid_score /= df_valid.shape[0]
print(f'score = {valid_score}')

# ################################################ inference 获取 loss ###############################################
df_test = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/test.csv')
model.eval()
submit_ids, submit_preds = [], []
for index in tqdm(range(df_test.shape[0])):
    columns = df_test.iloc[index].values
    scores = []
    input_ids = tokenizer(columns[1]+" <extra_id_0>", return_tensors="pt").input_ids.cuda()
    labels = tokenizer(["<extra_id_0> "+columns[2+p] for p in range(5)], return_tensors="pt", padding=True).input_ids
    minlen = np.min([len(l) for l in labels])
    for p in range(5):
        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=labels[p][:minlen].unsqueeze(0).cuda()).loss.detach().cpu().numpy()
        scores.append(float(loss))
    submit_ids.append(columns[0])
    submit_preds.append(scores)
# ############################################## inference 获取 probability ##########################################
from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

df_train = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/train.csv')
df_train = df_train.drop(columns="id")

df_train = pd.concat([
    df_train,
    pd.read_csv('/kaggle/input/additional-train-data-for-llm-science-exam/extra_train_set.csv'),
])
df_train.reset_index(inplace=True, drop=True)

deberta_v3_large = '/kaggle/input/deberta-v3-large-hf-weights'

tokenizer = AutoTokenizer.from_pretrained(deberta_v3_large)

dataset = Dataset.from_pandas(df_train)

options = 'ABCDE'
indices = list(range(5))

option_to_index = {option: index for option, index in zip(options, indices)}
index_to_option = {index: option for option, index in zip(options, indices)}


def preprocess(example):
    """The example is expected to be a dictionary with keys 'prompt', 'A', 'B', 'C', 'D', 'E', and 'answer'."""
    # The AutoModelForMultipleChoice class expects a set of question/answer pairs
    # so we'll copy our question 5 times before tokenizing
    first_sentence = [example['prompt']] * 5
    second_sentence = [example[option] for option in options]
    # Our tokenizer will turn our text into token IDs BERT can understand
    tokenized_example = tokenizer(first_sentence, second_sentence, truncation=True)
    tokenized_example['label'] = option_to_index[example['answer']]

    return tokenized_example


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch


tokenized_dataset = dataset.map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

training_args = TrainingArguments(
    warmup_ratio=0.8,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    report_to='none',
    output_dir='.'
)

model = AutoModelForMultipleChoice.from_pretrained(deberta_v3_large)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    train_dataset=tokenized_dataset,
)

trainer.train()

test_df = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/test.csv')
test_df.head()

# There are more verbose/elegant ways of doing this, but if we give our test set a random `answer` column
# we can make predictions directly with our trainer.
test_df['answer'] = 'A'

# Other than that we'll preprocess it in the same way we preprocessed test.csv
test_ds = Dataset.from_pandas(test_df)
tokenized_test_ds = test_ds.map(preprocess, batched=False, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

# Here we'll generate our "real" predictions on the test set
test_predictions = trainer.predict(tokenized_test_ds)

from sklearn.preprocessing import normalize
# 自回归模型产生的 loss 越小越好，自编码模型 inference 的 probability 越大越好
final_predictions = normalize(submit_preds)*0.2 + normalize(-test_predictions.predictions)*0.8

final_preds = [' '.join(np.array(list("ABCDE"))[np.argsort(s)][:3].tolist()) for s in final_predictions]

pd.DataFrame({'id': submit_ids, 'prediction': final_preds}).to_csv('submission.csv', index=False)
