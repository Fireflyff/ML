import pdfminer
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LlamaForCausalLM, AutoTokenizer
import torch


from langchain.document_loaders import pdf
# model_path = "/kaggle/input/gpt2model/gpt2"
# sentence_transformer_model = "moka-ai/m3e-base"
gpt_path = "/Users/yingying/Desktop/pre_train_model/gpt2"
glm_path = "/Users/yingying/Desktop/pre_train_model/THUDM_glm_large_chinese"
sentence_transformer_model = "/Users/yingying/Desktop/pre_train_model/m3e_model"


def read_pdf_langchain(file_path):
    loder = UnstructuredFileLoader(file_path)
    pages = loder.load()
    text = ""
    for i in pages:
        text += i.page_content
    text = text.replace("\n", "")
    raw_text = text.replace(" ", "")
    return raw_text


def read_pdf(file_path):
    fp = open(file_path, 'rb')

    # Create a PDF parser object associated with the file object
    parser = PDFParser(fp)

    # Create a PDF document object that stores the document structure.
    # Password for initialization as 2nd parameter
    document = PDFDocument(parser)
    # Check if the document allows text extraction. If not, abort.
    if not document.is_extractable:
        raise PDFTextExtractionNotAllowed

    # Create a PDF resource manager object that stores shared resources.
    rsrcmgr = PDFResourceManager()

    # BEGIN LAYOUT ANALYSIS.
    # Set parameters for analysis.
    laparams = LAParams(
        char_margin=10.0,
        line_margin=0.2,
        boxes_flow=0.2,
        all_texts=False,
    )
    # Create a PDF page aggregator object.
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # loop over all pages in the document
    text = ""
    for page in PDFPage.create_pages(document):
        # read the page into a layout object
        interpreter.process_page(page)
        layout = device.get_result()
        for obj in layout._objs:
            if isinstance(obj, pdfminer.layout.LTTextBoxHorizontal):
                text += obj.get_text()

    text = text.replace("\n", "")
    raw_text = text.replace(" ", "")
    return raw_text


def text_preprocess(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="。",
        chunk_size=70,
        chunk_overlap=0,
        length_function=len,
    )
    # 文本分割
    texts = text_splitter.split_text(raw_text)

    # 构建embedding
    embeddings = HuggingFaceEmbeddings(model_name=sentence_transformer_model)

    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch


def gpt_generation(checkpoint, text):
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = GPT2LMHeadModel.from_pretrained(checkpoint, trust_remote_code=True)
    model.eval()
    encoded_input = tokenizer.encode(text)
    answer_len = len(encoded_input)
    tokens_tensor = torch.tensor([encoded_input])

    stopids = tokenizer.convert_tokens_to_ids(["。"])[0]
    past = None
    for i in range(500):
        with torch.no_grad():

            output, past = model(tokens_tensor, past_key_values=past, return_dict=False)

        token = torch.argmax(output[..., -1, :])

        encoded_input += [token.tolist()]

        if stopids == token.tolist():
            break
        tokens_tensor = token.unsqueeze(0)

    return tokenizer.decode(encoded_input[answer_len:])


def glm_generation(checkpoint, text):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, trust_remote_code=True)
    # model = model.half().cuda()
    model.eval()

    # Inference
    prompt = text + " [gMASK]"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
    # inputs = inputs.to('cuda')
    outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id, no_repeat_ngram_size=1)
    return tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)[len(text):]


def ziya_generation(checkpoint, text):
    device = torch.device("cuda")
    model = LlamaForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    inputs = '<human>:' + text.strip() + '\n<bot>:'

    input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to(device)
    generate_ids = model.generate(
        input_ids,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.85,
        temperature=1.0,
        repetition_penalty=1.,
        eos_token_id=2,
        bos_token_id=1,
        pad_token_id=0)
    output = tokenizer.batch_decode(generate_ids)[0]
    return output


def chatbot(checkpoint, query, pdf_path):
    text = read_pdf_langchain(pdf_path)
    # 在这里添加聊天机器人的代码，使用输入的文本作为聊天机器人的输入，并返回答复文本
    docsearch = text_preprocess(text)
    docs = docsearch.similarity_search(query, k=3)
    prompt = ""
    # 构建prompt
    for doc in docs:
        prompt += doc.page_content + "\n"
        # prompt += doc.page_content
    prompt += query
    print(prompt)
    # result = gpt_generation(checkpoint, prompt)
    result = glm_generation(checkpoint, prompt)
    return result


# chatbot(model_path, "什么是幻觉？", "/kaggle/input/knowledge-base/_____.pdf")
print(chatbot(glm_path, "电力市场包括哪些含义？", "/Users/yingying/Desktop/homo_api/风险和信用管理7.2.docx"))
# print(read_pdf("/Users/yingying/Desktop/homo_api/电力现货市场基本规则.pdf"))


# reader = PdfReader("/Users/yingying/Desktop/homo_api/电力现货市场基本规则.pdf")
# raw_text = ''
# for i, page in enumerate(reader.pages):
#     text = page.extract_text()
#     if text:
#         raw_text += text
#
# print(raw_text)
