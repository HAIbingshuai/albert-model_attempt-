import jieba
import re
import string
from zhon.hanzi import punctuation
import os
import random
import copy

class Tokenize:
    def __init__(self,vocab_path = "./3500常用字.txt",stop_vocab_filepath = "./stopword.txt"):
        self.vocab_char_2_id_dict, self.vocab_id_2_char_dict = self.get_vocab(path=(vocab_path))
        self.cleaner = CleanChineseSentence(stop_vocab_filepath)

    #获取词袋的大小
    def get_vocab_size(self):
        vocab_length = len(self.vocab_char_2_id_dict)
        print("词袋的大小为：",vocab_length)
        return vocab_length

    # setence是中文文本的序列，由于需要清理，不要做成list，直接输入一个句子即可。
    # short_text = "我喜 欢的 seventinnn，几天去买几个吃的吧，大概10个左右again"
    def tokenizer(self,setence):
        setence = self.cleaner.cleaner(setence)
        setence_2_id_list = []
        setence = jieba.lcut(setence)
        for word in setence:
            if len(word) == 1:
                id = (self.vocab_char_2_id_dict.get(word,4))
                if id == 4:
                    pass
                else:
                    setence_2_id_list.append(id)
            else:
                #这里限制了中英文单词最多的长度使用7个
                #word_length = min(len(word),7)
                for i in range(len(word)):
                    if i == 0:
                        id = (self.vocab_char_2_id_dict.get(word[i], 4))
                        if id == 4:
                            pass
                        else:
                            setence_2_id_list.append(id)
                    else:
                        id = (self.vocab_char_2_id_dict.get(("##"+ word[i]), 4))
                        if id == 4:
                            pass
                        else:
                            setence_2_id_list.append(id)
        return setence_2_id_list

    #这里输入的是经过tokenizer函数处理过的数字序列，形如：
    #[2403, 2444, 1215, 836, 1277, 2291, 1986, 1677, 1277, 1058, 723, 836, 528, 802, 1029, 1058, 2995, 2723]
    #注意的是，这里没有加上[CLS]和[SEP]，也就是没加上1和2
    def transpose_id_2_char(self,setence_token_list):
        setence_list = []
        for id in setence_token_list:
            char = self.vocab_id_2_char_dict.get(id)
            setence_list.append(char)
        setence = "".join(setence_list)
        setence = re.sub("##","",setence)
        return setence

    def get_vocab(self,path):
        with open(path, encoding="UTF-8") as f:
            char_list = f.readline()
            vocab = []
            vocab_char_2_id_dict = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4}
            vocab_id_2_char_dict = {0: "[PAD]", 1: "[CLS]", 2: "[SEP]", 3: "[MASK]", 4: "[UNK]"}

            vocab_length = len(vocab_char_2_id_dict)
            # 这里留出了前500个字符用来添加一些不常用的东西
            for placeholder_id in range(500 - vocab_length):
                vocab_char_2_id_dict["placeholder_{}".format(placeholder_id)] = placeholder_id + vocab_length
                vocab_id_2_char_dict[placeholder_id + vocab_length] = "placeholder_{}".format(placeholder_id)

            vocab_counter = len(vocab_char_2_id_dict)
            for char in char_list[1:]:
                vocab.append(char)
                vocab_char_2_id_dict[char] = vocab_counter
                vocab_id_2_char_dict[vocab_counter] = char
                vocab_counter += 1

            # 这里是为下面的方法服务的
            # 新增的方法，输入一句话，返回一句经过处理的话: 为了支持中文全称mask，将被分开的词，将上特殊标记("#")，
            # 使得后续处理模块，能够知道哪些字是属于同一个词的。
            vocab_counter = len(vocab_char_2_id_dict)
            for char in char_list[1:]:
                char = "##" + char
                vocab.append(char)
                vocab_char_2_id_dict[char] = vocab_counter
                vocab_id_2_char_dict[vocab_counter] = char
                vocab_counter += 1

        return vocab_char_2_id_dict,vocab_id_2_char_dict

#文本清洗并不是适合所有的东西
class CleanChineseSentence:
    def __init__(self,vocab_file = "./stopword.txt"):
        self.vocab_list = self.获取停用词表(vocab_file)

    def cleaner(self,sentence):
        sentence = self.去除非汉字_数字_停用词(sentence)
        return sentence

    # 把非汉字的字符全部去掉
    def 去除非汉字_数字_停用词(self,sentence):
        sentence = sentence.strip()

        #1、把标点符号做了修正
        sentence =  self.英文标点符号转中文标点符号(sentence)

        #2、进行清洗，只保留字母、数字、汉字和标点符号(),.!?":
        sentence = self.remove_others(sentence)

        #3、删除多余的空白(including spaces, tabs, line breaks)'''
        sentence = self.remove_whitespaces(sentence)

        #3、删除英文字符，感觉也还是保留
        #sentence = re.compile(r'[a-zA-Z]+').sub('', sentence)

        #4、根据stopword.txt文件做去停用词
        splited_sentence = jieba.cut(sentence)
        sentence_list = []
        for word in splited_sentence:
            if word not in self.vocab_list:
                sentence_list.append(word)
        sentence = "".join(sentence_list)
        return sentence

    # 保留字母、数字、汉字和标点符号(),.!?":
    def remove_others(self,sentence):
        #下面是删除英文的处理代码
        return re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5()，。！？【】（）《》“‘]', ' ', sentence)

    # 删除多余的空白(including spaces, tabs, line breaks)'''
    def remove_whitespaces(self,sentence):
        sentence = sentence.replace('\r', '')
        sentence = sentence.replace('\t', ' ')
        sentence = sentence.replace('\f', ' ')
        sentence = sentence.replace(' ', ' ')
        return re.sub(r'\s{2,}', ' ', sentence)

    def 删除中英文标点(self,sentence):
        """删除字符串中的中英文标点.
        Args:
            sentence: 字符串
        """
        en_punc_tab = str.maketrans('', '', string.punctuation)  # ↓ ① ℃处理不了
        sent_no_en_punc = sentence.translate(en_punc_tab)
        return re.sub(r'[%s]+' % punctuation, "", sent_no_en_punc)

    # 删除句子列表中的空行，返回没有空行的句子列表
    def del_blank_lines(self,sentences):
        """删除句子列表中的空行，返回没有空行的句子列表
        Args:
            sentences: 字符串列表
        """
        return [s for s in sentences if s.split()]

    def 获取停用词表(self,vocab_file = "./很好的stopword.txt"):
        vocab_list = []
        for char in open(vocab_file,encoding="UTF-8").readlines():
            vocab_list.append(char.strip())
        return vocab_list

    def 英文标点符号转中文标点符号(self,string):
        E_pun = u',.!?[]()<>"\''
        C_pun = u'，。！？【】（）《》“‘'
        table = {ord(f): ord(t) for f, t in zip(E_pun, C_pun)}
        return string.translate(table)

#这里输入的document是一段话，即段中包含很多句话用逗号，或者句号。分割
#我的想法，段落都不需要
from albert_config import Config
def create_instances_from_document_albert(document,tokenize):
    setence_list = str(document).split("。")
    token_list = []
    for sentence in setence_list:
        if len(sentence) < 9:
            pass
        else:
            token = tokenize.tokenizer(sentence)
            token_list.append(token + [4096])

    # 这里主要是对a和b的顺序做一个交换，没有交换的是1，交换的是0
    lable = []  #作为标签集合
    train_token_list = []
    token_type_list = []

    #for i in range(random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),(len(token_list) - 1)):
    for i in range((len(token_list) - 1)):
        if random.random() >= 0.5:
            token_a = [1] + token_list[i] + [2]
            token_b = token_list[i + 1] + [2]

            token_type_list.append([0] * len(token_a) + [1] * len(token_b))
            train_token_list.append(token_a + token_b)
            lable.append(1)
        else:
            token_a = [1] + token_list[i] + [2]
            token_b = token_list[i + 1] + [2]

            token_type_list.append([0] * len(token_b) + [1] * len(token_a))
            train_token_list.append(token_b + token_a)
            lable.append(0)

    train_token_mask_list = []
    for _token in train_token_list:
        new_token = copy.deepcopy(_token)
        try:
            mask_list = random.sample(new_token, (int(len(new_token)*0.217)))
            for mask_id in mask_list:
                if mask_id == 1 or mask_id == 2:
                    pass
                else:
                    mask_index = new_token.index(mask_id)
                    new_token[mask_index] = 3  # 被替换成了mask
        except:print(new_token)
        train_token_mask_list.append(new_token)

    return train_token_mask_list,token_type_list,train_token_list,lable






if __name__ == "__main__":
    tokenize = Tokenize()
    document = "文本过滤和清理所涵盖的范围非常广泛，涉及文本解析和数据处理方面的问题。在非常简单的层次上，我们可能会用基本的字符串函数（例如str.upper()和str.lower()）将文本转换为标准形式。简单的替换操作可通过str.replace()或re.sub()来完成，它们把重点放在移除或修改特定的字符序列上。也可以利用unicodedata.normalize()来规范化文本。"
    train_token_mask_list,token_type_list,train_token_list,lable = create_instances_from_document_albert(document,tokenize)

    print(train_token_mask_list[0])
    print(train_token_list[0])
    # short_text = "我 喜欢的 seventinnn，几天去买几个吃的吧,大概10个左右again."
    # tokenize = Tokenize()
    # print(tokenize.get_vocab_size())
    # id_list = tokenize.tokenizer(short_text)
    # print(id_list)
    # setence = tokenize.transpose_id_2_char(id_list)
    # print(setence)