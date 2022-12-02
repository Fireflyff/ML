import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset

import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import DataCollatorWithPadding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CFG1:
    model = "microsoft/deberta-v3-base"
    path = "../input/0911-deberta-v3-base/"
    base = "../input/fb3models/microsoft-deberta-v3-base/"
    config_path = base + "config/config.json"
    print(config_path)
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    gradient_checkpointing = False
    batch_size = 24
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 10
    trn_fold = list(range(n_fold))
    num_workers = 4
    weight = 1.0


class CFG2:
    model = "microsoft/deberta-v3-large"
    path = "../input/0911-deberta-v3-large/"
    base = "../input/fb3models/microsoft-deberta-v3-large/"
    config_path = base + "config/config.json"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    gradient_checkpointing = False
    batch_size = 16
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 10
    trn_fold = list(range(n_fold))
    num_workers = 4
    weight = 1.0  ##


class CFG3:
    model = "microsoft/deberta-v2-xlarge"
    path = "../input/0911-deberta-v2-xlarge/"
    base = "../input/fb3models/microsoft-deberta-v2-xlarge/"
    config_path = base + "config/config.json"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    gradient_checkpointing = False
    batch_size = 4
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 10
    trn_fold = list(range(n_fold))
    num_workers = 4
    weight = 1.0


class CFG4:
    model = "microsoft/deberta-v3-base"
    path = "../input/0913-deberta-v3-base-fgm/"
    base = "../input/fb3models/microsoft-deberta-v3-base/"
    config_path = base + "config/config.json"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    gradient_checkpointing = False
    batch_size = 24
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 10
    trn_fold = list(range(n_fold))
    num_workers = 4
    weight = 1.0


class CFG5:
    model = "microsoft/deberta-v3-large"
    path = "../input/0914-deberta-v3-large-fgm/"
    base = "../input/fb3models/microsoft-deberta-v3-large/"
    config_path = base + "config/config.json"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    gradient_checkpointing = False
    batch_size = 16
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 10
    trn_fold = list(range(n_fold))
    num_workers = 4
    weight = 1.0  ##


class CFG6:
    model = "microsoft/deberta-v2-xlarge"
    path = "../input/0919-deberta-v2-xlarge/"
    base = "../input/fb3models/microsoft-deberta-v2-xlarge/"
    config_path = base + "config/config.json"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    gradient_checkpointing = False
    batch_size = 4
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 10
    trn_fold = list(range(n_fold))
    num_workers = 4
    weight = 1.0  #


class CFG7:
    model = "microsoft/deberta-v2-xlarge-mnli"
    path = "../input/0919-deberta-v2-xlarge-mnli/"
    base = "../input/fb3models/microsoft-deberta-v2-xlarge/"
    config_path = base + "config/config.json"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    gradient_checkpointing = False
    batch_size = 4
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 10
    trn_fold = list(range(n_fold))
    num_workers = 4
    weight = 3.0  ###


class CFG8:
    model = "microsoft/deberta-v3-large"
    path = "../input/0925-deberta-v3-large-unscale/"
    base = "../input/fb3models/microsoft-deberta-v3-large/"
    config_path = base + "config/config.json"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    gradient_checkpointing = False
    batch_size = 16
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 10
    trn_fold = list(range(n_fold))
    num_workers = 4
    weight = 1.2  ##


class CFG9:
    model = "microsoft/deberta-v3-large"
    path = "../input/0926-deberta-v3-large-unscale/"
    base = "../input/fb3models/microsoft-deberta-v3-large/"
    config_path = base + "config/config.json"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    gradient_checkpointing = False
    batch_size = 16
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 10
    trn_fold = list(range(n_fold))
    num_workers = 4
    weight = 3.0  #


class CFG10:
    model = "microsoft/deberta-v3-large"
    path = "../input/0927-deberta-v3-large-unscale/"
    base = "../input/fb3models/microsoft-deberta-v3-large/"
    config_path = base + "config/config.json"
    tokenizer = AutoTokenizer.from_pretrained(base + 'tokenizer/')
    gradient_checkpointing = False
    batch_size = 16
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 10
    trn_fold = list(range(n_fold))
    num_workers = 4
    weight = 1.0  # 17


# CFG_list = [CFG1, CFG2, CFG3, CFG4, CFG5, CFG6, CFG7, CFG8, CFG9, CFG10]
CFG_list = [CFG5, CFG8, CFG9, CFG10]


# ====================================================
# Utils
# ====================================================
def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = mean_squared_error(y_true, y_pred, squared=False)  # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores


def get_logger(filename='inference'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = get_logger()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)

# ====================================================
# oof
# ====================================================
for CFG in CFG_list:
    oof_df = pd.read_pickle(CFG.path+'oof_df.pkl')
    labels = oof_df[CFG.target_cols].values
    preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
    score, scores = get_score(labels, preds)
    LOGGER.info(f'Model: {CFG.model} Score: {score:<.4f}  Scores: {scores}')

# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        #max_length=CFG.max_len,
        #pad_to_max_length=True,
        #truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs


# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            LOGGER.info(self.config)
        else:
            self.config = AutoConfig.from_pretrained(config_path, output_hidden_states=True)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.pool = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 6)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(feature)
        return output

# ====================================================
# inference
# ====================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

for _idx, CFG in enumerate(CFG_list):
    test = pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')
    submission = pd.read_csv('../input/feedback-prize-english-language-learning/sample_submission.csv')
    # sort by length to speed up inference
    test['tokenize_length'] = [len(CFG.tokenizer(text)['input_ids']) for text in test['full_text'].values]
    test = test.sort_values('tokenize_length', ascending=True).reset_index(drop=True)

    test_dataset = TestDataset(CFG, test)
    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             collate_fn=DataCollatorWithPadding(tokenizer=CFG.tokenizer, padding='longest'),
                             num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    predictions = []
    for fold in CFG.trn_fold:
        model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
        state = torch.load(CFG.path+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                           map_location=torch.device('cpu'))
        print(CFG.path+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
        model.load_state_dict(state['model'])
        prediction = inference_fn(test_loader, model, device)
        predictions.append(prediction)
        del model, state, prediction; gc.collect()
        torch.cuda.empty_cache()
    predictions = np.mean(predictions, axis=0)
    test[CFG.target_cols] = predictions
    submission = submission.drop(columns=CFG.target_cols).merge(test[['text_id'] + CFG.target_cols], on='text_id', how='left')
    submission[['text_id'] + CFG.target_cols].to_csv(f'submission_{_idx + 1}.csv', index=False)
    del test, submission, predictions, test_dataset, test_loader; gc.collect()
    torch.cuda.empty_cache()

test = pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')
submission = pd.read_csv('../input/feedback-prize-english-language-learning/sample_submission.csv')

sub1 = pd.read_csv(f'submission_1.csv')[CFG1.target_cols] * CFG1.weight
sub2 = pd.read_csv(f'submission_2.csv')[CFG2.target_cols] * CFG2.weight
sub3 = pd.read_csv(f'submission_3.csv')[CFG3.target_cols] * CFG3.weight
sub4 = pd.read_csv(f'submission_4.csv')[CFG4.target_cols] * CFG4.weight
sub5 = pd.read_csv(f'submission_5.csv')[CFG5.target_cols] * CFG5.weight
sub6 = pd.read_csv(f'submission_6.csv')[CFG6.target_cols] * CFG6.weight
sub7 = pd.read_csv(f'submission_7.csv')[CFG7.target_cols] * CFG7.weight
sub8 = pd.read_csv(f'submission_8.csv')[CFG8.target_cols] * CFG8.weight
sub9 = pd.read_csv(f'submission_9.csv')[CFG9.target_cols] * CFG9.weight
sub10 = pd.read_csv(f'submission_10.csv')[CFG10.target_cols] * CFG10.weight

ens = ((sub1 + sub2 + sub3 + sub4 + sub5 + sub6 + sub7 + sub8 + sub9 + sub10)
       /(CFG1.weight + CFG2.weight + CFG3.weight + CFG4.weight + CFG5.weight
         + CFG6.weight + CFG7.weight + CFG8.weight + CFG9.weight + CFG10.weight))
# ens = (( sub1 + sub2 + sub3 + sub4)
#        /( CFG1.weight + CFG2.weight + CFG3.weight + CFG4.weight))


submission[CFG1.target_cols] = ens
submission.to_csv('submission.csv', index=False)