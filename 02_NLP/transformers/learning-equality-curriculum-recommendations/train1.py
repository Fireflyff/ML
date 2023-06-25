import sys
sys.path.append('/kaggle/input/sentencetransformers/sentence-transformers')

import pandas as pd
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch
tqdm.pandas()

BASE_DIR = '/kaggle/input/learning-equality-curriculum-recommendations'

content_data = pd.read_csv(os.path.join(BASE_DIR, 'content.csv'))
correlations_data = pd.read_csv(os.path.join(BASE_DIR, 'correlations.csv'))
topics_data = pd.read_csv(os.path.join(BASE_DIR, 'topics.csv'))
sub_data = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))

model = SentenceTransformer('/kaggle/input/sentenceembeddingmodels/0_Transformer', device='cuda')

sub_ids = sub_data['topic_id'].tolist()
topics_data_ = topics_data.query(f'id in {sub_ids}').reset_index(drop=True)

topics_data_['merged_text'] = topics_data_['title'].fillna('')+','+topics_data_['description'].fillna('')
content_data['merged_text'] = content_data['title'].fillna('')+','+content_data['description'].fillna('')


topics_data_['merged_text'].fillna('Not Available', inplace=True)
content_data['merged_text'].fillna('Not Available', inplace=True)

t_data = topics_data_[['id', 'merged_text']]
c_data = content_data[['id', 'merged_text']]

corpus_embeddings = model.encode(c_data['merged_text'], convert_to_tensor=True)

pred = []
top_k = 5
for query in t_data['merged_text']:
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    pred.append(top_results[1].cpu().numpy())

pred_final = []
for idx in pred:
    pid = c_data['id'][idx]
    pred_final.append(' '.join(pid))

sub_data.loc[:, 'content_ids'] = pred_final
sub_data.to_csv('submission.csv', index=False)
sub_data.head()
