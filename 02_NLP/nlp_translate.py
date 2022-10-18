from google_trans_new import google_translator

text = ["苍兰诀很好看", "苍兰诀剧情很搞笑", "鞋子穿起来很舒服", "这个鸡蛋很亮", "哇偶！", "姑姑也想过过过过过的生活。"]


translator = google_translator()
translators = translator.translate(text, 'en')
print(translators)

zh_trans = translator.translate(translators, 'zh-CN')
print(zh_trans)

ko_trans = translator.translate(zh_trans, 'ko')
print(ko_trans)

print(translator.translate(ko_trans, 'zh-CN'))
