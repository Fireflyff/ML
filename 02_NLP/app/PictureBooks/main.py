from PIL import Image, ImageDraw, ImageFont
import openai
import json
import requests
import os
import sys
import re
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape
import time
openai.api_key = "input_your_API_KEY"


def from_prompt_to_story(prompt):
    user_prompt = f'''
    请根据下面的提示词写一篇分为6段的儿童故事，每一段大约100字:
    {prompt}
    ******
    You should only respond in JSON format as described below
    Response format:
    {{
        "story":"story"
    }},
    '''
    jason_messages = [
        {"role": "user", "content": user_prompt}]

    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-4",  # 只能使用gpt-4,不要用gpt-3.5-turbo
                messages=jason_messages,
                temperature=0
            )
            break
        except Exception as e:
            print(e)
            time.sleep(2)  # 等待 2 秒后重试

    return json.loads(completion.choices[0].message['content'])['story']


def split_story(story):
    sentences = [line.strip() for line in story.split('\n') if line.strip()]
    return sentences


def add_prefix_to_story(sentences):
    for sentence in sentences:
        numbered_list = [f"第{i}段：{item}" for i,
                         item in enumerate(sentences, start=1)]
    # print(numbered_list)
    return numbered_list


def extract_keywords(story_list):
    sentences = '\n'.join(story_list)
    # print(sentences)
    user_prompt = f'''
    请提取下面每段中的关键词,每一段都要提取关键词,将关键词展示为英文：
    {sentences}
    ******
    您应该仅以 JSON 格式响应，如下所述
    响应格式：
    [{{
        "keywords":"keyword1,keyword2,..."
    }},...]
    '''
    jason_messages = [
        {"role": "user", "content": user_prompt}]

    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=jason_messages,
                temperature=0
            )
            break
        except Exception as e:
            print(e)
            time.sleep(2)  # 等待 2 秒后重试

    return json.loads(completion.choices[0].message['content'])


def generate_img(keywords):
    # print(keywords)
    if not os.path.exists('./img/'):
        os.makedirs('./img/')
    for i, keyword in enumerate(keywords):
        while True:
            try:
                res = openai.Image.create(
                    prompt=keyword['keywords'] +
                    ", children's picture book style",
                    n=1,
                    size="1024x1024"
                )
                break
            except Exception as e:
                print(e)
                time.sleep(2)  # 等待 2 秒后重试

        url = res['data'][0]['url']
        # 发送GET请求并获取响应
        response = requests.get(url)
        # 保存图片到本地
        with open('./img/'+str(i+1)+'.png', 'wb') as f:
            f.write(response.content)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def get_img_files(dir_path):
    current_path = os.getcwd()  # 获取当前路径
    folder_path = os.path.join(current_path, dir_path)
    """递归获取指定文件夹下的所有文件"""
    files = []
    # 遍历文件夹下的所有文件和文件夹
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # 判断是否是文件，如果是文件则添加到列表中
        if os.path.isfile(item_path):
            files.append(item_path)
        # 如果是文件夹，则递归获取该文件夹下的所有文件
        elif os.path.isdir(item_path):
            files.extend(list_all_files(item_path))

    sorted_files = sorted(
        files, key=lambda x: natural_sort_key(os.path.basename(x)))
    return sorted_files


def add_text_to_img(story_list):
    img_list = get_img_files('img')

    if not os.path.exists('./CompletedIMG/'):
        os.makedirs('./CompletedIMG/')

    for k, img_node in enumerate(img_list):
        # 打开图片
        image = Image.open(img_node)

        # 创建一个绘图对象
        draw = ImageDraw.Draw(image)

        # 设置文字内容和字体
        text = story_list[k]
        font = ImageFont.truetype('simsun.ttc', 36, encoding='unic')

        # 设置文字位置和颜色
        x, y = 10, 10
        color = 'rgb(255, 68, 100)'

        text_list = [text[i:i+20] for i in range(0, len(text), 20)]
        for line in text_list:
            # 在图片上绘制文字
            draw.text((x, y), line, fill=color, font=font)
            y += font.getsize(line)[1] + 10
        # 保存修改后的图片
        image.save('./CompletedIMG/'+str(k+1)+'.jpg')


def add_image_to_pdf():
    image_paths = get_img_files('CompletedIMG')
    # 创建一个新的PDF文件
    c = canvas.Canvas('story.pdf')
    for image_path in image_paths:
        # 打开图片并获取其大小
        img = Image.open(image_path)
        img_width, img_height = img.size

        # 创建一个新的页面，大小为图片大小
        c.setPageSize((img_width, img_height))

        # 将图片添加到PDF文件中
        c.drawImage(image_path, 0, 0, img_width, img_height)

        # 创建一个新的页面
        c.showPage()

    # 保存PDF文件
    c.save()


def main(prompt):
    story = from_prompt_to_story(prompt)
    story_list = split_story(story)
    prefixstory_list = add_prefix_to_story(story_list)
    keywords = extract_keywords(prefixstory_list)
    generate_img(keywords)
    add_text_to_img(story_list)
    add_image_to_pdf()


if __name__ == '__main__':
    test_prompt = '精灵 星空 友谊'
    main(test_prompt)
