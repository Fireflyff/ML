import datasets
# 加载数据
dataset = datasets.load_from_disk("../../NLP_datasets/ChnSentiCorp")
print(dataset)