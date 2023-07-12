from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
tokenizer = GPT2Tokenizer.from_pretrained('/Users/yingying/Desktop/pre_train_model/gpt2')
model = GPT2LMHeadModel.from_pretrained('/Users/yingying/Desktop/pre_train_model/gpt2')
model.eval()

# PromptTemplate
"""
from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["prompt_key", "year_day"],
    template="请到系统查询一下截止今天我的{prompt_key},{year_day}?",
)
print(prompt.format(prompt_key="剩余调休" year_day="年假信息"))
"""
# text = "Based on the following known information, provide concise and professional answers to user questions. " \
#        "Adding fabricated elements to the answer is not allowed.Known content:This short sleeve is suitable for people weighing 115-170." \
#        "Question:Can people weighing 167 wear this short sleeve?"

text = "PageRank is an algorithm that measures the importance of webpages based on the links pointing to them. " \
       "The basic idea is that authoritative pages get more links. So pages with more links should rank higher in search results. " \
       "Especially if those links come from popular pages (i.e., pages that have high PageRank scores themselves)." \
       "Previously, SEOs could see the PageRank score of any webpage via the Google Toolbar." \
       "Answer the question based on known information:what is pagerank?"
# text = "the color of the clothes is Black." \
#        "the sizes of the clothes: 0-1, 2-3, 9-10." \
#        "the target audience for clothing is Women." \
#        "the material of the clothes is Silk."\
#        "let's think step by step:  what is the color of the clothes?"
encoded_input = tokenizer.encode(text)
answer_len = len(encoded_input)
tokens_tensor = torch.tensor([encoded_input])

stopids = tokenizer.convert_tokens_to_ids(["."])[0]
past = None
for i in range(100):
    with torch.no_grad():
        output, past = model(tokens_tensor, past_key_values=past, return_dict=False)

    token = torch.argmax(output[..., -1, :])

    encoded_input += [token.tolist()]

    if stopids == token.tolist():
        break
    tokens_tensor = token.unsqueeze(0)

sequence = tokenizer.decode(encoded_input[answer_len:])

print(sequence)
