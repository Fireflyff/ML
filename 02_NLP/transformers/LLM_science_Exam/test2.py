from datasets import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from typing import Optional, Union
from dataclasses import dataclass
# device = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoTokenizer, DebertaV2Model
import torch

# Let's import training data and have a look
import pandas as pd
import numpy as np

train_df = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/train.csv')
train_df = train_df.drop(columns="id")


df_train = pd.concat([
    train_df,
    pd.read_csv('/kaggle/input/additional-train-data-for-llm-science-exam/extra_train_set.csv'),
])
df_train.reset_index(inplace=True, drop=True)

# creating a hugging face dataset
train_ds = Dataset.from_pandas(df_train)

model_dir = '/kaggle/input/deberta-v3-large-hf-weights'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# model =DebertaV2ForMultipleChoice.from_pretrained(model_dir)
model = AutoModelForMultipleChoice.from_pretrained(model_dir)

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


tokenized_train_ds = train_ds.map(preprocess, batched=False,
                                  remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if 'label' in features[0].keys() else 'labels'
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


training_args = TrainingArguments(
    warmup_ratio=0.8,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    report_to='none',
    output_dir='.'
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    train_dataset=tokenized_train_ds,
)

# Training should take about a minute
trainer.train()


# The following function gets the indices of the highest scoring answers for each row
# and converts them back to our answer format (A, B, C, D, E)
def predictions_to_map_output(predictions):
    sorted_answer_indices = np.argsort(-predictions)
    top_answer_indices = sorted_answer_indices[:, :3]  # Get the first three answers in each row
    # vectorize the the option as a no we get to alphabet eg 0-A,1-B,etc.
    top_answers = np.vectorize(index_to_option.get)(top_answer_indices)
    return np.apply_along_axis(lambda row: ' '.join(row), 1, top_answers)


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

# Now we can create our submission using the id column from test.csv
submission_df = test_df[['id']]
submission_df['prediction'] = predictions_to_map_output(test_predictions.predictions)


submission_df.to_csv('submission.csv', index=False)
