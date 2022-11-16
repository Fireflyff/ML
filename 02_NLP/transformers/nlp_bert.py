# https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html
import torch
from transformers import BertTokenizer
from transformers import BertForMaskedLM


PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定繁简中文 BERT-BASE预训练模型
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
# vocab = tokenizer.vocab
# print("字典大小：", len(vocab))
# 字典大小： 21128
# random_tokens = random.sample(list(vocab), 10)
# random_ids = [vocab[t] for t in random_tokens]
# print("{0:20}{1:15}".format("token", "index"))
# print("-" * 30)
# for t, id in zip(random_tokens, random_ids):
#     print("{0:15}{1:10}".format(t, id))
"""
token               index          
------------------------------
kicstart2           12614
##℃                  8320
姫                    2009
滸                    4019
##薹                 19020
##mp                 9085
##✿                 13640
##巅                 15387
100g                10606
舀                    5641
"""

text = "[CLS] 等到潮水 [MASK] 了，就知道誰沒穿褲子。"
tokens = tokenizer.tokenize(text)
# ['[CLS]', '等', '到', '潮', '水', '[MASK]', '了', '，', '就', '知'....]
ids = tokenizer.convert_tokens_to_ids(tokens)
# [101, 5023, 1168, 4060, 3717, 103, 749, 8024, 2218, 4761....]
"""
除了一般的 wordpieces 以外，BERT 裡頭有 5 個特殊 tokens 各司其職：

[CLS]：在做分類任務時其最後一層的 repr. 會被視為整個輸入序列的 repr.
[SEP]：有兩個句子的文本會被串接成一個輸入序列，並在兩句之間插入這個 token 以做區隔
[UNK]：沒出現在 BERT 字典裡頭的字會被這個 token 取代
[PAD]：zero padding 遮罩，將長度不一的輸入序列補齊方便做 batch 運算
[MASK]：未知遮罩，僅在預訓練階段會用到
"""
tokens_tensor = torch.tensor([ids])  # (1, seq_len)
segments_tensors = torch.zeros_like(tokens_tensor)  # (1, seq_len)
maskedLM_model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME)

# 使用masked LM 估计 [MASK] 位置所代表的实际token
maskedLM_model.eval()
with torch.no_grad():
    outputs = maskedLM_model(tokens_tensor, segments_tensors)
    predictions = outputs[0]  # (1, seq_len, num_hidden_units)

del maskedLM_model

# 将 [mask] 位置的概率分布取 top k 最有可能的 tokens 出来
masked_index = 5
k = 3
probs, indices = torch.topk(torch.softmax(predictions[0, masked_index], -1), k)
prediction_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())

print("输入 tokens：", tokens[:10], "...")
print("-" * 50)
for i, (t, p) in enumerate(zip(prediction_tokens, probs), 1):
    tokens[masked_index] = t
    print("Top {} ({:2}%):{}".format(i, int(p.item() * 100), tokens[:10]), "...")
