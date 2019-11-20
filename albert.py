import tensorflow as tf
from Albert_untils import albert_until
from albert_config import Config
import numpy as np

class EmbeddingLookupFactorized(tf.keras.layers.Layer):
    def __init__(self,vocab_size = Config.vocab_size,hidden_size = Config.hidden_size,embdding_size = Config.embdding_size):
        super(EmbeddingLookupFactorized, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embdding_size = embdding_size

        # 为该层创建一个可训练的权重
        self.weight_embedding_table = tf.Variable(
                initial_value=(tf.random.truncated_normal(shape=[self.vocab_size, self.embdding_size])))
        self.project_variable = tf.Variable(
            initial_value=(tf.random.truncated_normal(shape=[self.embdding_size, self.hidden_size])))

    def call(self, inputs):
        input_ids = inputs

        input_shape = tf.shape(input_ids)
        input_shape_as_list = input_ids.shape.as_list()

        input_ids = tf.cast(input_ids,tf.int32)
        input_ids_batch_size, input_ids_sequene_length = input_shape[0],input_shape_as_list[1]

        input_ids = tf.expand_dims(input_ids, axis=2)
        flat_input_ids = tf.reshape(input_ids, [-1])

        output_middle = tf.gather(self.weight_embedding_table,flat_input_ids)

        output = tf.matmul(output_middle, self.project_variable)

        output = tf.reshape(output, (input_ids_batch_size,input_ids_sequene_length,self.hidden_size))
        return output

    def get_embedding_table(self):
        return self.weight_embedding_table,self.project_variable

class Albert:
    def __init__(self,trainable = True):
        #with tf.variable_creator_scope(trainable = self.trainable):
        # word_index, embeddings_matrix = get_embedding_table.get_embedding_model_pretrained_vector()#这里是使用了一些预训练的词汇embedding
        self.embedding_table = EmbeddingLookupFactorized()

        self.token_type_embedding_table = tf.keras.layers.Embedding(input_dim=2,output_dim=Config.hidden_size)

        self.position_embedding = albert_until.positional_encoding(Config.max_length,Config.hidden_size)

        self.embedding_layer_norm = tf.keras.layers.LayerNormalization()

        self.encoder_layer = albert_until.AlbertEncoderLayer()

        #下面是做训练时需要的东西,获取了embdding_factorized中生成的2个embedding_table
        self.weight_embedding_table,self.project_variable = self.embedding_table.get_embedding_table()

        #这个是为了做next_setence做的取出,注意这里的激活函数
        self.pooled_output_dense = tf.keras.layers.Dense(2,activation=tf.nn.tanh)

    def __call__(self,inputs):
        #获取输入值
        word_token_inputs,token_type_ids = inputs
        """
            这里只使用的 词token + 相互存在 + pos，mask是以 词token 为基础做的，结果挺好的！
            这里的token_type_ids是对前后2个句子进行分别，[【cls】,seq_1,seq_1,seq_1,【sep】,seq_2,seq_2,seq_2,【sep】]
            被表示成[0,0,0,0,0,1,1,1,1],注意连接的位置，特别是sep的位置
        """
        #获取输入token的维度
        token_shape = tf.shape(word_token_inputs)
        batch_size = token_shape[0]
        seq_length = token_shape[1]

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
        token_type_embedding = self.token_type_embedding_table(token_type_ids)

        word_embedding = self.embedding_table(word_token_inputs)

        position_embedding = tf.slice(self.position_embedding,[0,0,0],[-1,seq_length,-1])

        embedding = word_embedding + position_embedding + token_type_embedding
        embedding = self.embedding_layer_norm(embedding)
        embedding = tf.keras.layers.Dropout(0.1)(embedding)

        word_token_mask = albert_until.create_mutual_padding_mask(word_token_inputs, word_token_inputs)

        #Todo 这里输出的是一个3D的矩阵，[batch_size,sequence_length,hidden_size]
        albert_sequence_output = self.encoder_layer([embedding,word_token_mask])

        "下面开始做训练部分，这里的训练有2个，mask_language_model和next_setence_model"

        #Todo 1、mask_language_model,这里经过了2次乘积，将重新进组合成[batch_size,setence_length,vocab_size]的形式
        middle_output = tf.matmul(albert_sequence_output, self.project_variable, transpose_b=True)
        mask_language_model_logits = tf.matmul(middle_output, self.weight_embedding_table, transpose_b=True)

        #Todo 2、下面是next_tence_model,取出第一个作整个句子的代表,注意这里的是albert_sequence_output
        #下面把pooling换成GlobalMaxPool1D做测试,感觉这个比较快
        albert_pooled_output = tf.keras.layers.GlobalMaxPool1D()(albert_sequence_output)
        albert_pooled_output = self.pooled_output_dense(albert_pooled_output)


        return albert_pooled_output

if __name__ == "__main__":
    word_token_inputs = tf.keras.Input(shape=(Config.max_length,))
    token_type_ids = tf.keras.Input(shape=(Config.max_length,))

    mask_language_model_logits,albert_pooled_output = Albert()([word_token_inputs, token_type_ids])
    model = tf.keras.Model((word_token_inputs, token_type_ids),(albert_pooled_output))
    print(model.summary())
