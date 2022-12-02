# https://github.com/lansinuote/Huggingface_Toturials
from transformers import BertTokenizer
# 加载预训练字典和分词方法
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path="bert-base-chinese",
    cache_dir=None,
    force_download=False
)
sents = ["选择珠江花园的原因就是方便。",
         "笔记本的键盘确实爽。",
         "房间太小。其他的都一般。",
         "今天才知道这书还有第6卷，真有点郁闷。",
         "机器背面似乎被撕了张什么标签，残胶还在。"]

# out = tokenizer.encode(text=sents[0], text_pair=sents[1],
#                        # 当句子长度大于max_length时，截断
#                        truncation=True,
#                        # 一律补pad到max_length长度
#                        padding="max_length",
#                        max_length=30,
#                        return_tensors=None
#                        )
# out ==> [101, 6848, 2885, 4403, 3736, 5709, 1736, 4638, 1333, 1728, 2218, 3221, 3175, 912, 511, 102,
# 5011, 6381, 3315, 4638, 7241, 4669, 4802, 2141, 4272, 511, 102, 0, 0, 0]
# tokenizer.decode(out) ==>
# [CLS] 选 择 珠 江 花 园 的 原 因 就 是 方 便 。 [SEP] 笔 记 本 的 键 盘 确 实 爽 。 [SEP] [PAD] [PAD] [PAD]
# todo:encode_plus
# out1 = tokenizer.encode_plus(text=sents[0], text_pair=sents[1],
#                              # 当句子长度大于max_length时，截断
#                              truncation=True,
#                              # 一律补pad到max_length长度
#                              padding="max_length",
#                              max_length=30,
#                              add_special_tokens=True,
#                              # 可取值tf(tensorflow),pt(pytorch),np(numpy),默认返回list
#                              return_tensors=None,
#                              # 返回token_type_ids
#                              return_token_type_ids=True,
#                              # 返回attention_mask
#                              return_attention_mask=True,
#                              # 返回special_tokens_mask特殊符号标识
#                              return_special_tokens_mask=True,
#                              # 返回offset_mapping标识每个词的起止位置，这个参数只能在BertTokenizerFast中使用
#                              # return_offsets_mapping=True,
#                              # 返回length标识长度
#                              return_length=True)
# input_ids : [101, 6848, 2885, 4403, 3736, 5709, 1736, 4638, 1333, 1728, 2218, 3221, 3175, 912, 511, 102,
# 5011, 6381, 3315, 4638, 7241, 4669, 4802, 2141, 4272, 511, 102, 0, 0, 0]
# token_type_ids : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
# special_tokens_mask : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
# attention_mask : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
# length : 30
# tokenizer.decode(out1["input_ids"]) ==>
# [CLS] 选 择 珠 江 花 园 的 原 因 就 是 方 便 。 [SEP] 笔 记 本 的 键 盘 确 实 爽 。 [SEP] [PAD] [PAD] [PAD]
# todo:batch_encode_plus_1
# out2 = tokenizer.batch_encode_plus(batch_text_or_text_pairs=[sents[0], sents[1]],
#                                    # 当句子长度大于max_length时，截断
#                                    truncation=True,
#                                    # 一律补pad到max_length长度
#                                    padding="max_length",
#                                    max_length=15,
#                                    add_special_tokens=True,
#                                    # 可取值tf(tensorflow),pt(pytorch),np(numpy),默认返回list
#                                    return_tensors=None,
#                                    # 返回token_type_ids
#                                    return_token_type_ids=True,
#                                    # 返回attention_mask
#                                    return_attention_mask=True,
#                                    # 返回special_tokens_mask特殊符号标识
#                                    return_special_tokens_mask=True,
#                                    # 返回offset_mapping标识每个词的起止位置，这个参数只能在BertTokenizerFast中使用
#                                    # return_offsets_mapping=True,
#                                    # 返回length标识长度
#                                    return_length=True)
# input_ids : [[101, 6848, 2885, 4403, 3736, 5709, 1736, 4638, 1333, 1728, 2218, 3221, 3175, 912, 102],
# [101, 5011, 6381, 3315, 4638, 7241, 4669, 4802, 2141, 4272, 511, 102, 0, 0, 0]]
# token_type_ids : [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
# special_tokens_mask : [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]]
# length : [15, 12]
# attention_mask : [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
# tokenizer.decode(out2["input_ids"][0]) ==> [CLS] 选 择 珠 江 花 园 的 原 因 就 是 方 便 [SEP]
# tokenizer.decode(out2["input_ids"][1]) ==> [CLS] 笔 记 本 的 键 盘 确 实 爽 。 [SEP] [PAD] [PAD] [PAD]
# todo:batch_encode_plus_2
# out3 = tokenizer.batch_encode_plus(batch_text_or_text_pairs=[(sents[0], sents[1]), (sents[2], sents[3])],
#                                    # 当句子长度大于max_length时，截断
#                                    truncation=True,
#                                    # 一律补pad到max_length长度
#                                    padding="max_length",
#                                    max_length=30,
#                                    add_special_tokens=True,
#                                    # 可取值tf(tensorflow),pt(pytorch),np(numpy),默认返回list
#                                    return_tensors=None,
#                                    # 返回token_type_ids
#                                    return_token_type_ids=True,
#                                    # 返回attention_mask
#                                    return_attention_mask=True,
#                                    # 返回special_tokens_mask特殊符号标识
#                                    return_special_tokens_mask=True,
#                                    # 返回offset_mapping标识每个词的起止位置，这个参数只能在BertTokenizerFast中使用
#                                    # return_offsets_mapping=True,
#                                    # 返回length标识长度
#                                    return_length=True)
# input_ids : [[101, 6848, 2885, 4403, 3736, 5709, 1736, 4638, 1333, 1728, 2218, 3221, 3175, 912, 511, 102,
# 5011, 6381, 3315, 4638, 7241, 4669, 4802, 2141, 4272, 511, 102, 0, 0, 0],
# [101, 2791, 7313, 1922, 2207, 511, 1071, 800, 4638, 6963, 671, 5663, 511, 102,
# 791, 1921, 2798, 4761, 6887, 6821, 741, 6820, 3300, 5018, 127, 1318, 8024, 4696, 3300, 102]]
# token_type_ids : [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
# special_tokens_mask : [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
# [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
# length : [27, 30] (去掉padding的句子长度)
# attention_mask : [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
# tokenizer.decode(out3["input_ids"][0]) ==>
# [CLS] 选 择 珠 江 花 园 的 原 因 就 是 方 便 。 [SEP] 笔 记 本 的 键 盘 确 实 爽 。 [SEP] [PAD] [PAD] [PAD]

#################################################################################################################
# for k, v in out3.items():
#     print(k, ":", v)
# print(tokenizer.decode(out3["input_ids"][0]))


# 添加新词
tokenizer.add_tokens(new_tokens=["XX"])
# 添加新符号
tokenizer.add_special_tokens({'eos_token': '[EOS]'})
zidian = tokenizer.get_vocab()
print(zidian)
print(tokenizer)
# {'[PAD]': 0, '[unused1]': 1, '[unused2]': 2, '[unused3]': 3, ...}
