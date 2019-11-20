from char_untils import tokenizer
from tqdm import tqdm
import os

#这个函数主要是用作对大段的txt文件进行处理
def get_data(cleaner,folder_path = "C:/NLP数据集/审计数据集/"):
    file_path_list = os.listdir(folder_path)
    document = []
    for file_path in file_path_list:
        with open(folder_path + file_path,encoding="UTF-8") as f:
            context = f.readlines()
            for line in tqdm(context):
                if len(line) > 9:
                    if line == '\n':
                        line = line.strip("\n")
                    line = line.strip().replace(" ","")
                    line = cleaner.cleaner(line)
                    document.append(line)
    document = "".join(document)
    return document



if __name__ =="__main__":
    cleaner = tokenizer.CleanChineseSentence("./stopword.txt")

    document = get_data(cleaner)
    setence_list = document.split("。")

    for seq in setence_list:
        print(seq)









