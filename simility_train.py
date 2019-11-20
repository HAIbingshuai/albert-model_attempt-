from char_untils import tokenizer
import csv
from tqdm import tqdm
from fintunning_config import  Config

tokenize = tokenizer.Tokenize("./char_untils/3500常用字.txt", "./char_untils/stopword.txt")
cleaner = tokenizer.CleanChineseSentence("./char_untils/stopword.txt")

token_list = []
token_type_list = []
label_list = []

with open('./chip/train.tsv', encoding="UTF-8")as f:
    f_csv = f.readlines()
    for row in tqdm(f_csv):
        row = str(row).strip().split("	")
        question_1 = row[0]
        question_2 = row[1]
        label = int(row[2])

        question_1 = cleaner.cleaner(question_1)
        question_2 = cleaner.cleaner(question_2)

        question_1_token = [1] + tokenize.tokenizer(question_1) + [2]
        question_2_token = tokenize.tokenizer(question_2) + [2]
        question_1_type = [0] * len(question_1_token)
        question_2_type = [1] * len(question_2_token)

        token_list.append(question_1_token + question_2_token)
        token_type_list.append(question_1_type + question_2_type)
        label_list.append(label)

dev_token_list = []
dev_token_type_list = []
dev_label_list = []

with open('./chip/dev.tsv', encoding="UTF-8")as f:
    f_csv = f.readlines()
    for row in tqdm(f_csv):
        row = str(row).strip().split("	")
        question_1 = row[0]
        question_2 = row[1]
        label = int(row[2])

        question_1 = cleaner.cleaner(question_1)
        question_2 = cleaner.cleaner(question_2)

        question_1_token = [1] + tokenize.tokenizer(question_1) + [2]
        question_2_token = tokenize.tokenizer(question_2) + [2]
        question_1_type = [0] * len(question_1_token)
        question_2_type = [1] * len(question_2_token)

        dev_token_list.append(question_1_token + question_2_token)
        dev_token_type_list.append(question_1_type + question_2_type)
        dev_label_list.append(label)

import numpy as np
from char_untils import char_until
token_list = char_until.pad_sequences(token_list,maxlen=Config.max_length)
token_type_list = char_until.pad_sequences(token_type_list,maxlen=Config.max_length)
label_list = np.array(label_list)

dev_token_list = char_until.pad_sequences(token_list,maxlen=Config.max_length)
dev_token_type_list = char_until.pad_sequences(token_type_list,maxlen=Config.max_length)
dev_label_list = np.array(label_list)

def generator(batch_size = 32):
    batch_num = len(label_list)//batch_size
    while 1:
        for i in range(batch_num):
            start = batch_size * i
            end = batch_size * (i + 1)

            yield (token_list[start:end],token_type_list[start:end]),(label_list[start:end])


if __name__ == "__main__":
    import tensorflow as tf
    from fintunning_config import  Config
    import simility_vs_albert as simility_model

    with tf.device("/GPU:0"):
        token_input = tf.keras.Input(shape=(Config.max_length,))
        token_type_input = tf.keras.Input(shape=(Config.max_length,))
        simility_output = simility_model.Simility()([token_input,token_type_input])
        model = tf.keras.Model((token_input,token_type_input),(simility_output))

        callback = tf.keras.callbacks.ModelCheckpoint(filepath="./simility_saver/model", period=1, save_weights_only=True,verbose=2)
        model.compile(tf.optimizers.Adam(1e-5), loss=tf.losses.sparse_categorical_crossentropy, metrics=["accuracy"])
        batch_size = 768

        model.fit_generator(generator(batch_size), epochs=50, callbacks=[callback],steps_per_epoch=(len(label_list)) // batch_size,verbose=2,validation_data=((dev_token_list,dev_token_type_list),(dev_label_list)))
        model.save_weights("./simility_saver/model")



