一、使用说明：
1、安装依赖：pip install -r requirements.txt
2、在key.py 填入openai_key
3、main.py中files_arr代表文件名，使用之前需要在files创建一个user_id目录，用于存放当前user的文件，例如当user_id为jason时，就需要在files下创建jason目录，即/files/jason/
add_files(files_arr,user_id),加载文档,files_arr与user_id同上
delete_files(files_arr, 'jason'),删除某个文档,files_arr与user_id同上
user_chat(question, user_id),与文档聊天，question:用户问题,user_id同上

二、可能会遇到的异常
文档过短或者过长都会报错
解决方案：
1、当文档过短时,可以选择不要分块,documents = loader.load()直接将documents中的内容直接向chatgpt提问即可
2、当文档过长时,可以限制总字数
以上两种方案都需要去测总字数的下限以及上限
