from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  # 解决跨域问题
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import uvicorn, json
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # 允许跨域的源列表，例如 ["http://www.example.org"] 等等，["*"] 表示允许任何源
    allow_origins=["*"],
    # 跨域请求是否支持 cookie，默认是 False，如果为 True，allow_origins 必须为具体的源，不可以是 ["*"]
    allow_credentials=False,
    # 允许跨域请求的 HTTP 方法列表，默认是 ["GET"]
    allow_methods=["*"],
    # 允许跨域请求的 HTTP 请求头列表，默认是 []，可以使用 ["*"] 表示允许所有的请求头
    # 当然 Accept、Accept-Language、Content-Language 以及 Content-Type 总之被允许的
    allow_headers=["*"],
    # 可以被浏览器访问的响应头, 默认是 []，一般很少指定
    # expose_headers=["*"]
    # 设定浏览器缓存 CORS 响应的最长时间，单位是秒。默认为 600，一般也很少指定
    # max_age=1000
)


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    text = json_post_list.get('prompt')
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

    return sequence


if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('/Users/yingying/Desktop/pre_train_model/gpt2')
    model = GPT2LMHeadModel.from_pretrained('/Users/yingying/Desktop/pre_train_model/gpt2')
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
