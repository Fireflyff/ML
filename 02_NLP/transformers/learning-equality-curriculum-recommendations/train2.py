import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from transformers import AutoTokenizer, AutoModel
import cupy as cp
from cuml.metrics import pairwise_distances

device = "cuda" if torch.cuda.is_available() else "cpu"

import os
for dirname, _, filenames in os.walk('/kaggle/input/learning-equality-curriculum-recommendations'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


class CFG:
    INPUT = '/kaggle/input/learning-equality-curriculum-recommendations'
    MODEL = '/kaggle/input/sentence-embedding-models/paraphrase-MiniLM-L12-v2'
    MAX_LEN = 384
    SELECT_TOP_N = 5


content_df = pd.read_csv(f'{CFG.INPUT}/content.csv')
correlations_df = pd.read_csv(f'{CFG.INPUT}/correlations.csv')
topics_df = pd.read_csv(f'{CFG.INPUT}/topics.csv')
sub_df = pd.read_csv(f'{CFG.INPUT}/sample_submission.csv')

model = AutoModel.from_pretrained(CFG.MODEL)
model.eval()
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL)

from tqdm.auto import tqdm

vecs = []
for _, row in tqdm(content_df.iterrows(), total=len(content_df)):
    title = row['title']
    if type(title) is float:
        title = row['description']
    if type(title) is float:
        title = row['text']

    tok = tokenizer(title)
    for k, v in tok.items():
        tok[k] = torch.tensor(v[:CFG.MAX_LEN]).to(device).unsqueeze(0)
    with torch.no_grad():
        output = model(**tok)
    vec = output.last_hidden_state.squeeze(0).mean(0).cpu()
    vecs.append(vec)

vecs1 = torch.stack(vecs)

sub_topic_ids = sub_df['topic_id'].tolist()
_topics_df = topics_df.query(f'id in {sub_topic_ids}')

vecs = []
for _, row in tqdm(_topics_df.iterrows(), total=len(_topics_df)):
    title = row['title']
    if type(title) is float:
        title = row['description']
    if type(title) is float:
        title = "This content contains no text."

    tok = tokenizer(title)
    for k, v in tok.items():
        tok[k] = torch.tensor(v[:CFG.MAX_LEN]).to(device).unsqueeze(0)
    with torch.no_grad():
        output = model(**tok)
    vec = output.last_hidden_state.squeeze(0).mean(0).cpu()
    vecs.append(vec)

vecs2 = torch.stack(vecs)

vecs1 = cp.asarray(vecs1)
vecs2 = cp.asarray(vecs2)

predicts = []
for v2 in vecs2:
    sim = pairwise_distances(v2.reshape(1, len(v2)), vecs1, metric='cosine')
    p = " ".join([content_df.loc[s, 'id'] for s in sim.argsort(1)[0, :CFG.SELECT_TOP_N].get()])
    predicts.append(p)

sub_df['content_ids'] = predicts

sub_df.to_csv('submission.csv', index=None)

