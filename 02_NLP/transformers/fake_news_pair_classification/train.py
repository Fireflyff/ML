"""
将处理后的数据转换成 BERT 格式，并声称3个tensors
- tokens_tensor：两个句子合并后的索引序列，包含[CLS]和[SEP]
- segments_tensor：用来识别两个句子界限的 binary tensor
_ label_tensor：将分类标签转换成类别索引的 tensor ，如果是测试集则为 None
"""
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification


class FakeNewsDataset(Dataset):
    # 读取预处理的数据，并初始化一些参数
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test"]
        self.mode = mode
        self.df = pd.read_csv(mode + ".csv", sep="\t").fillna("")
        self.len = len(self.df)
        self.label_map = {"agreed": 0, "disagreed": 1, "unrelated": 2}
        self.tokenizer = tokenizer  # 这里使用 BERT tokenizer

    def __getitem__(self, idx):
        if self.mode == "test":
            text_a, text_b = self.df.iloc[idx, :2].values
            label_tensor = None
        else:
            text_a, text_b, label = self.df.iloc[idx, :].values
            # 将 label 文字也转化成索引方便转成tensor
            label_id = self.label_map[label]
            label_tensor = torch.tensor(label_id)

        # 建立第一个句子的 BERT tokens 并加入分隔符 [SEP]
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)
        # 第二个句子的 BERT tokens
        tokens_b = self.tokenizer.tokenize(text_b)
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a
        # 将整个 token 序列转成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        # 将第一句包含 [SEP] 的 token 位置设为 0， 其他为 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.long)
        return tokens_tensor, segments_tensor, label_tensor

    def __len__(self):
        return self.len


# 初始化一个专门读取训练样本的 Dataset，使用中文 BERT 断词
PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定繁簡中文 BERT-BASE 預訓練模型
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
trainset = FakeNewsDataset("train", tokenizer=tokenizer)


"""
实现返回 mini-batch 的 DataLoader
- tokens_tensors   : (batch_size, max_seq_len_in_batch)
- segments_tensors : (batch_size, max_seq_len_in_batch)
- masks_tensors    : (batch_size, max_seq_len_in_batch)
- label_ids        : (batch_size)
"""


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None

    # zero pad 到同一序列长度
    # 以 list 中长度最长 tensor 为基准进行填充
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    # attention masks,将 tokens_tensors 里面不为 zero padding
    # 的位置设为 1 让 BERT 只关注这些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

# 初始化一个每次回传 64 个训练样本的 DataLoader
# 利用 collate_fn 将 list of samples 合成一个 mini-batch
BATCH_SIZE = 64
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)
NUM_LABELS = 3
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
