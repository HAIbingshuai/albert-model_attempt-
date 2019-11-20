import csv
from config import Config
import numpy as np
from model import cleaner_中文句子清洗 as han_cleaner
import random
cleaner = han_cleaner.CleanChineseSentence("./model/很好的stopword.txt")
from Albert_untils import char_untils

from model import 使用预训练的词向量 as get_embedding_table
# word_index, embeddings_matrix = get_embedding_table.get_embedding_model_pretrained_vector()
word_index, word_vector, embeddings_matrix = get_embedding_table.get_embedding_model()




def train_generator(batch_size = 96):
    def process_seq_mask(seq_list):
        seq_mask = np.not_equal(seq_list, 0)
        seq_min_num = sum(np.min(seq_mask, axis=0))
        if seq_min_num > 4:
            mask_list = range(seq_min_num)
            seed = random.choice(mask_list[1:-1])

            for seq in seq_list:
                seq[seed] = random.choice(range(1, 20000))
                seq[seed + 1] = random.choice(range(1, 20000))
            return seq_list
        else:
            return seq_list


    question_1_list = []
    question_2_list = []
    question_1_exist_token_list = []
    question_2_exist_token_list = []
    label_list = []
    disease_list = []

    with open('./chip/train.tsv', encoding="UTF-8")as f:
        # f_csv = csv.reader(f)
        f_csv = f.readlines()
        for row in f_csv:
            row = row.strip()
            row = row.split("	")

            question_1 = row[0]
            question_2 = row[1]
            label = int(row[2])
            # disease = row[3]

            question_1 = cleaner.cleaner(question_1)
            question_2 = cleaner.cleaner(question_2)

            question_1_list.append(question_1)
            question_2_list.append(question_2)
            label_list.append(label)
            disease_list.append("[UNK_title]")

            (setence_1_exist_token), (setence_2_exist_token) = char_untils.比较2个句子中存在情况(question_1, question_2)
            question_1_exist_token_list.append(setence_1_exist_token)
            question_2_exist_token_list.append(setence_2_exist_token)

    with open('./chip/train.csv', encoding="UTF-8")as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            question_1 = row[0]
            question_2 = row[1]
            label = int(row[2])
            disease = row[3]

            question_1 = cleaner.cleaner(question_1)
            question_2 = cleaner.cleaner(question_2)

            question_1_list.append(question_1)
            question_2_list.append(question_2)
            label_list.append(label)
            disease_list.append(disease)

            (setence_1_exist_token), (setence_2_exist_token) = char_untils.比较2个句子中存在情况(question_1, question_2)
            question_1_exist_token_list.append(setence_1_exist_token)
            question_2_exist_token_list.append(setence_2_exist_token)


    question_1_char_token_list = get_embedding_table.tokenizer_char(question_1_list, word_index, disease_list)
    question_1_word_token_list = get_embedding_table.tokenizer_word(question_1_list, word_index, disease_list)

    question_2_char_token_list = get_embedding_table.tokenizer_char(question_2_list, word_index, disease_list)
    question_2_word_token_list = get_embedding_table.tokenizer_word(question_2_list, word_index, disease_list)

    question_1_char_token_list = char_untils.pad_sequences(question_1_char_token_list, maxlen=Config.max_length)
    question_1_word_token_list = char_untils.pad_sequences(question_1_word_token_list, maxlen=Config.max_length)
    question_2_char_token_list = char_untils.pad_sequences(question_2_char_token_list, maxlen=Config.max_length)
    question_2_word_token_list = char_untils.pad_sequences(question_2_word_token_list, maxlen=Config.max_length)
    question_1_exist_token_list = char_untils.pad_sequences(question_1_exist_token_list, maxlen=Config.max_length)
    question_2_exist_token_list = char_untils.pad_sequences(question_2_exist_token_list, maxlen=Config.max_length)
    label_list = char_untils.one_hot(label_list, depth=2)

    batch_num = len(label_list)//batch_size
    while 1:
        seed = int(np.random.random()*100)
        np.random.seed(seed);np.random.shuffle(question_1_char_token_list)
        np.random.seed(seed);np.random.shuffle(question_1_word_token_list)
        np.random.seed(seed);np.random.shuffle(question_2_char_token_list)
        np.random.seed(seed);np.random.shuffle(question_2_word_token_list)
        np.random.seed(seed);np.random.shuffle(question_1_exist_token_list)
        np.random.seed(seed);np.random.shuffle(question_2_exist_token_list)
        np.random.seed(seed);np.random.shuffle(label_list)

        for i in range(batch_num):
            start = batch_size*i
            end = batch_size*(i + 1)

            if random.random() >= 0.217:
                yield ((question_1_char_token_list[start:end]),
                       (question_1_word_token_list[start:end]),
                       (question_2_char_token_list[start:end]),
                       (question_2_word_token_list[start:end]),
                       question_1_exist_token_list[start:end],
                       question_2_exist_token_list[start:end]) \
                    , (label_list[start:end])
            else:
                yield (process_seq_mask(question_1_char_token_list[start:end]),
                       process_seq_mask(question_1_word_token_list[start:end]),
                       process_seq_mask(question_2_char_token_list[start:end]),
                       process_seq_mask(question_2_word_token_list[start:end]),
                       question_1_exist_token_list[start:end],
                       question_2_exist_token_list[start:end])\
                        ,(label_list[start:end])


question_1_val_list = []
question_2_val_list = []
question_1_val_exist_token_list = []
question_2_val_exist_token_list = []
label_val_list = []
disease_val_list = []

with open('./chip/dev.tsv',encoding="UTF-8")as f:
    #f_csv = csv.reader(f)
    f_csv = f.readlines()
    for row in f_csv:
        row = row.strip()
        row = row.split("	")

        question_1 = row[0]
        question_2 = row[1]
        label = int(row[2])
        #disease = row[3]

        question_1 = cleaner.cleaner(question_1)
        question_2 = cleaner.cleaner(question_2)

        question_1_val_list.append(question_1)
        question_2_val_list.append(question_2)
        label_val_list.append(label)
        disease_val_list.append("[UNK_title]")

        (setence_1_exist_token), (setence_2_exist_token) = char_untils.比较2个句子中存在情况(question_1, question_2)
        question_1_val_exist_token_list.append(setence_1_exist_token)
        question_2_val_exist_token_list.append(setence_2_exist_token)

question_1_char_val_token_list = get_embedding_table.tokenizer_char(question_1_val_list,word_index,disease_val_list)
question_1_word_val_token_list = get_embedding_table.tokenizer_word(question_1_val_list,word_index,disease_val_list)

question_2_char_val_token_list = get_embedding_table.tokenizer_char(question_2_val_list,word_index,disease_val_list)
question_2_word_val_token_list = get_embedding_table.tokenizer_word(question_2_val_list,word_index,disease_val_list)

question_1_char_val_token_list = char_untils.pad_sequences(question_1_char_val_token_list,maxlen=Config.max_length)
question_1_word_val_token_list = char_untils.pad_sequences(question_1_word_val_token_list,maxlen=Config.max_length)
question_2_char_val_token_list = char_untils.pad_sequences(question_2_char_val_token_list,maxlen=Config.max_length)
question_2_word_val_token_list = char_untils.pad_sequences(question_2_word_val_token_list,maxlen=Config.max_length)
question_1_val_exist_token_list = char_untils.pad_sequences(question_1_val_exist_token_list,maxlen=Config.max_length)
question_2_val_exist_token_list = char_untils.pad_sequences(question_2_val_exist_token_list,maxlen=Config.max_length)
label_val_list = char_untils.one_hot(label_val_list,depth=2)

"--------------------test----------------------"

question_1_test_list = []
question_2_test_list = []
question_1_test_exist_token_list = []
question_2_test_exist_token_list = []
label_test_list = []
disease_test_list = []

with open('./chip/test.tsv',encoding="UTF-8")as f:
    #f_csv = csv.reader(f)
    f_csv = f.readlines()
    for row in f_csv:
        row = row.strip()
        row = row.split("	")

        question_1 = row[0]
        question_2 = row[1]
        label = int(row[2])
        #disease = row[3]

        question_1 = cleaner.cleaner(question_1)
        question_2 = cleaner.cleaner(question_2)

        question_1_test_list.append(question_1)
        question_2_test_list.append(question_2)
        label_test_list.append(label)
        disease_test_list.append("[UNK_title]")

        (setence_1_exist_token), (setence_2_exist_token) = char_untils.比较2个句子中存在情况(question_1, question_2)
        question_1_test_exist_token_list.append(setence_1_exist_token)
        question_2_test_exist_token_list.append(setence_2_exist_token)

question_1_char_test_token_list = get_embedding_table.tokenizer_char(question_1_test_list,word_index,disease_test_list)
question_1_word_test_token_list = get_embedding_table.tokenizer_word(question_1_test_list,word_index,disease_test_list)

question_2_char_test_token_list = get_embedding_table.tokenizer_char(question_2_test_list,word_index,disease_test_list)
question_2_word_test_token_list = get_embedding_table.tokenizer_word(question_2_test_list,word_index,disease_test_list)

question_1_char_test_token_list = char_untils.pad_sequences(question_1_char_test_token_list,maxlen=Config.max_length)
question_1_word_test_token_list = char_untils.pad_sequences(question_1_word_test_token_list,maxlen=Config.max_length)
question_2_char_test_token_list = char_untils.pad_sequences(question_2_char_test_token_list,maxlen=Config.max_length)
question_2_word_test_token_list = char_untils.pad_sequences(question_2_word_test_token_list,maxlen=Config.max_length)
question_1_test_exist_token_list = char_untils.pad_sequences(question_1_test_exist_token_list,maxlen=Config.max_length)
question_2_test_exist_token_list = char_untils.pad_sequences(question_2_test_exist_token_list,maxlen=Config.max_length)
label_test_list = char_untils.one_hot(label_test_list,depth=2)
