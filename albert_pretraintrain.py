from char_untils import get_dataset
from char_untils import tokenizer
import random
from albert_config import Config
import numpy as np
from char_untils import char_until
import tensorflow as tf
import albert

def get_albert_pretrain_model(saved_checkpoint_filepath="./saver/model",load_sequence_output = False):
    word_token_inputs = tf.keras.Input(shape=(Config.max_length,))
    token_type_ids = tf.keras.Input(shape=(Config.max_length,))

    albert_pooled_output = albert.Albert()([word_token_inputs, token_type_ids])
    model = tf.keras.Model((word_token_inputs, token_type_ids), (albert_pooled_output))

    #model.load_weights(saved_checkpoint_filepath)

    if load_sequence_output:

        new_model = tf.keras.Model(model.inputs,outputs=model.get_layer('albert_encoder_layer').output)
        return new_model
    else:
        return model

if __name__ == "__main__":

    tokenize =tokenizer.Tokenize("./char_untils/3500常用字.txt","./char_untils/stopword.txt")
    cleaner = tokenizer.CleanChineseSentence("./char_untils/stopword.txt")
    document =get_dataset.get_data(cleaner)

    train_token_mask_list,token_type_list,train_token_list,lable =tokenizer.create_instances_from_document_albert(document,tokenize)
    train_token_mask_list = char_until.pad_sequences(train_token_mask_list,Config.max_length)
    token_type_list = char_until.pad_sequences(token_type_list,Config.max_length)
    train_token_list = char_until.pad_sequences(train_token_list,Config.max_length)
    lable = np.array(lable)
    print(len(lable))

    def generator(batch_size = 24):

        while 1:
            train_token_mask_list, token_type_list, train_token_list, lable = tokenizer.create_instances_from_document_albert(
                document, tokenize)
            train_token_mask_list = char_until.pad_sequences(train_token_mask_list, Config.max_length)
            token_type_list = char_until.pad_sequences(token_type_list, Config.max_length)
            train_token_list = char_until.pad_sequences(train_token_list, Config.max_length)
            lable = np.array(lable)

            batch_num = len(lable) // batch_size
            for i in range(batch_num):
                strat_point = batch_size * i
                end_point = batch_size * (i + 1)

                yield (train_token_mask_list[strat_point:end_point],token_type_list[strat_point:end_point]),(lable[strat_point:end_point])

    with tf.device("/GPU:0"):

        model = get_albert_pretrain_model()

        callback = tf.keras.callbacks.ModelCheckpoint(filepath="./saver/model", period=5, save_weights_only=True, verbose=2)
        model.compile(tf.keras.optimizers.Adam(1e-5),loss= tf.losses.sparse_categorical_crossentropy,metrics=["accuracy"])

        batch_size = 26
        model.fit_generator(generator(batch_size), epochs=10240000, verbose=2,steps_per_epoch=len(lable) // batch_size,callbacks=[callback])



