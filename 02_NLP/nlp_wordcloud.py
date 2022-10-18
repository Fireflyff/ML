from itertools import chain

import jieba.posseg as pseg
import matplotlib.pyplot as plt
import numpy
from PIL import Image

from wordcloud import WordCloud
import pandas as pd


def get_a_list(text):
    r = []
    for g in pseg.lcut(text):
        if g.flag == 'a':
            r.append(g.word)

    return r


def get_word_clond(keywords_list):
    # mask_pic = numpy.array(Image.open("/Users/yingying/Desktop/tests/NLP_data/love.jpeg"))
    mask_pic = numpy.array(Image.open("/Users/yingying/Desktop/picture/cat1.jpeg"))
    print(mask_pic.shape)
    wordcloud = WordCloud(max_words=100,
                          background_color='white',
                          font_path="/Users/yingying/Desktop/tests/NLP_data/simfang.ttf",
                          mask=mask_pic)
    keywords_string = " ".join(keywords_list)
    wordcloud.generate(keywords_string)
    wordcloud.to_file("苍兰诀点评.png")

    print(wordcloud)
    plt.figure()
    plt.imshow(wordcloud, interpolation=None)
    plt.axis("off")
    plt.show()


p_train_data = pd.read_csv("/Users/yingying/Desktop/tests/NLP_data/苍兰诀点评.csv")

data_list = list(chain(*map(lambda x: get_a_list(x), p_train_data.iloc[:, 0])))
print(data_list)
get_word_clond(data_list)



