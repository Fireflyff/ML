import os
import sys
import json
from data_space import *
from chat_document import *
from store_documents import *


def add_files(files_name, user_id):
    for file_name in files_name:
        # 获取文件后缀
        file_extension = file_name.split(".")[-1]
        match file_extension:
            case "pdf":
                return store_pdf(file_name, user_id)
            case "docx":
                return store_word(file_name, user_id)
            case "txt":
                return store_text(file_name, user_id)
            case "pptx":
                return store_pptx(file_name, user_id)
            case _:
                return False


def delete_files(files_name, user_id):
    try:
        for file_name in files_name:
            file_path = os.path.join(sys.path[0], 'files', user_id, file_name)
            os.remove(file_path)  # 删除文件
            delete_from_vdb(user_id, file_name)
            delete_from_json(user_id, file_name)
        return True
    except OSError as e:
        print(f"Error: {file_path} - {e.strerror}.")
        return False


def user_chat(question, user_id):
    return chat(question, user_id)


if __name__ == '__main__':
    files_arr = ['xxxx.pdf']
    # add_files(files_arr, 'jason')
    # print(user_chat('大致内容是什么', 'jason')['answer'])
    # delete_files(files_arr, 'jason')
