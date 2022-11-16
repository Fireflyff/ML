import os
import pandas as pd
data_path = "../../../NLP_datasets/fake-news-pair-classification-challenge"
df_train = pd.read_csv(f"{data_path}/train.csv")

empty_title = ((df_train["title2_zh"].isnull())
               | (df_train["title1_zh"].isnull())
               | (df_train["title2_zh"] == '')
               | (df_train["title2_zh"] == '0'))
df_train = df_train[~empty_title]

# 除掉过长的样本
MAX_LENGTH = 30
df_train = df_train[~(df_train.title1_zh.apply(lambda x: len(x)) > MAX_LENGTH)]
df_train = df_train[~(df_train.title2_zh.apply(lambda x: len(x)) > MAX_LENGTH)]

# 只用 1% 训练数据看看 BERT 对少量标注数据有多少帮助
SAMPLE_FRAC = 0.01
df_train = df_train.sample(frac=SAMPLE_FRAC, random_state=9527)

df_train = df_train.reset_index()
df_train = df_train.loc[:, ["title1_zh", "title2_zh", "label"]]
df_train.columns = ["text_a", "text_b", "label"]
df_train.to_csv("train.csv", sep="\t", index=False)
print("训练样本数：", len(df_train))
print(df_train.head())
print(df_train.label.value_counts() / len(df_train))

df_test = pd.read_csv(f"{data_path}/test.csv")
df_test = df_test.loc[:, ["title1_zh", "title2_zh", "id"]]
df_test.columns = ["text_a", "text_b", "Id"]
df_test.to_csv("test.csv", sep="\t", index=False)
print("预测样本数：", len(df_test))
print(df_test.head())
print("测试样本数/训练样本数：{:.1f}".format(len(df_test) / len(df_train)))
