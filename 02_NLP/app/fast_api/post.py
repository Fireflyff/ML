import requests
import json
headers = {'Content-Type': 'application/json'}
prompt = "PageRank is an algorithm that measures the importance of webpages based on the links pointing to them. " \
         "The basic idea is that authoritative pages get more links. So pages with more links should rank higher in search results. " \
         "Especially if those links come from popular pages (i.e., pages that have high PageRank scores themselves)." \
         "Previously, SEOs could see the PageRank score of any webpage via the Google Toolbar.?" \
         "question: what is pagerank?"
         # "Answer the question based on known information:what is pagerank?"\



data = json.dumps({
  'prompt': prompt,
})
# print("ChatGLM prompt:",prompt)
# 调用api
response = requests.post("http://192.168.113.143:8000/", headers=headers, data=data)
print(response.json())