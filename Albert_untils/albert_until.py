import tensorflow as tf
import math
import numpy as np
from albert_config import Config
# Todo embedding全部叠加以后要做个layerNorm

class AlbertEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, albert_attention_n_layer=Config.n_layer, hidden_size=Config.hidden_size,
                 do_return_all_layers="sequence_output",name="AlbertEncoder_xiaohua"):
        super(AlbertEncoderLayer, self).__init__()
        self.albert_attention_n_layer = albert_attention_n_layer
        self.hidden_size = hidden_size
        self.do_return_all_layers = do_return_all_layers

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        self.albert_multi_head_attention = AlbertMultiHeadAttentionLayer()

        self.output_layer_dense = tf.keras.layers.Dense(self.hidden_size,kernel_initializer=create_attention_layer_initializer())
        self.output_layer_norm = tf.keras.layers.LayerNormalization()

        self.output_intermediate = tf.keras.layers.Dense(self.hidden_size,kernel_initializer=create_attention_layer_initializer(),activation=gelu)

        self.output_dense = tf.keras.layers.Dense(self.hidden_size,kernel_initializer=create_attention_layer_initializer())
        self.output_norm = tf.keras.layers.LayerNormalization()

        #这里的pooled_out目前在我这里不需要，保留只是为了确认写法规范
        # self.pooled_output_dense = tf.keras.layers.Dense(self.hidden_size,kernel_initializer=create_attention_layer_initializer(),
        #                                                  activation=tf.keras.activations.tanh)

        super(AlbertEncoderLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        encoder_embedding, encoder_mask = inputs  # 这里的maks需要注意,这里是3D的，由下面算create_attention_mask_from_input_mask出来

        input_shape = tf.shape(encoder_embedding)#或者写成 encoder_embedding.get_shape().as_list() or tf.shape(encoder_embedding)
        input_shape_as_list = encoder_embedding.shape.as_list()

        prev_output = encoder_embedding
        # 由于这里是一样的，所以就只使用一个encoder_embedding作为quer,key,value
        # 这里的原来是3D的被转成2D，在最后输出的时候转换回来
        all_layer_outputs = []
        for i in range(self.albert_attention_n_layer):
            layer_input = prev_output
            attention_output = self.albert_multi_head_attention([layer_input, layer_input, layer_input, encoder_mask])

            attention_output = self.output_layer_dense(attention_output)
            attention_output = tf.keras.layers.Dropout(0.1)(attention_output)
            attention_output = self.output_layer_norm(attention_output + layer_input)

            intermediate_output = self.output_intermediate(attention_output)
            layer_output = self.output_dense(intermediate_output)
            layer_output = tf.keras.layers.Dropout(0.1)(layer_output)
            layer_output = self.output_norm(layer_output + attention_output)
            prev_output = layer_output
            all_layer_outputs.append(layer_output)

        if self.do_return_all_layers == "all_layer_outputs":  # 全部输出
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = reshape_2D_tensor_to_3D_matrix(layer_output, batch_size=input_shape[0],
                                                              sequence_length=input_shape_as_list[1],
                                                              hidden_size=input_shape_as_list[2])
                final_outputs.append(final_output)
            return final_outputs

        elif self.do_return_all_layers == "sequence_output":  # 只输出最后一层的3D
            sequence_output = reshape_2D_tensor_to_3D_matrix(prev_output, batch_size=input_shape[0],
                                                             sequence_length=input_shape_as_list[1],
                                                             hidden_size=input_shape_as_list[2])
            return sequence_output

        # else:  # 输出最后一层的第一个切片，用来代替全部
        #     sequence_output = reshape_2D_tensor_to_3D_matrix(prev_output, batch_size=input_shape[0],
        #                                                      sequence_length=input_shape_as_list[1],
        #                                                      hidden_size=input_shape_as_list[2])
        #     first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        #     pooled_output = self.pooled_output_dense(first_token_tensor)
        #     return pooled_output


class AlbertMultiHeadAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, n_head=Config.n_head, hidden_size=Config.hidden_size):
        super(AlbertMultiHeadAttentionLayer, self).__init__()
        self.n_head = n_head
        self.hidden_size = hidden_size

    def build(self, input_shape):

        self.query_2D_dense_layer = tf.keras.layers.Dense(self.hidden_size,
                                                          kernel_initializer=create_attention_layer_initializer())
        self.key_2D_dense_layer = tf.keras.layers.Dense(self.hidden_size,
                                                        kernel_initializer=create_attention_layer_initializer())
        self.value_2D_dense_layer = tf.keras.layers.Dense(self.hidden_size,
                                                          kernel_initializer=create_attention_layer_initializer())

        super(AlbertMultiHeadAttentionLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, inputs):
        # 这里虽然输进来3个，但是只用到query和key，写上value是和传统相符
        query, key, value, mask = inputs  # 这里的maks需要注意,这里是3D的，由下面算create_attention_mask_from_input_mask出来

        query_shape = tf.shape(query)

        batch_size = query_shape[0]
        sequence_length = query_shape[1]

        query_2d = reshape_3D_embedding_to_2D_matrix(query)
        key_2d = reshape_3D_embedding_to_2D_matrix(key)

        query_layer = self.query_2D_dense_layer(query_2d)
        key_layer = self.query_2D_dense_layer(key_2d)
        value_layer = self.query_2D_dense_layer(key_2d)

        query_layer = self.transpose_2D_tensor_to_4D(query_layer, batch_size, sequence_length)
        key_layer = self.transpose_2D_tensor_to_4D(key_layer, batch_size, sequence_length)
        value_layer = self.transpose_2D_tensor_to_4D(value_layer, batch_size, sequence_length)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(self.hidden_size / self.n_head)))

        if mask is not None:
            # 对于mask的标定，使用 1 代表有内容，而0代表被padded的位置
            adder_mask = (tf.cast(mask, tf.float32)) * -10000.0
            attention_scores += adder_mask

        attention_probs = tf.nn.softmax(attention_scores)
        attention_probs = tf.keras.layers.Dropout(0.1)(attention_probs)

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

        context_layer = tf.reshape(
                context_layer,[-1, sequence_length, self.hidden_size])

        return context_layer

    def transpose_2D_tensor_to_4D(self, input_tensor, batch_size, seq_length):
        output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, self.n_head, (self.hidden_size // self.n_head)])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor


def reshape_3D_embedding_to_2D_matrix(input_tensor, hidden_size=Config.hidden_size):  # 这里是将3D变成了2D
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""

    output_tensor = tf.reshape(input_tensor, [-1, hidden_size])
    return output_tensor

def create_attention_layer_initializer():
    return tf.keras.initializers.TruncatedNormal(stddev=0.02)

def create_attention_mask_from_input_mask(query_tensor, key_input_token,name = "mask"):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    query_shape = tf.shape(query_tensor)
    batch_size = query_shape[0]
    query_seq_length = query_shape[1]

    key_shape = tf.shape(key_input_token)
    key_seq_length = key_shape[1]

    key_mask = tf.cast(tf.reshape(key_input_token, [batch_size, 1, key_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(shape=[batch_size, query_seq_length, 1],name=name)

    # Here we broadcast along two dimensions to create the mask.
    mask = tf.keras.layers.Lambda(lambda x:tf.multiply(x[0],x[1]))([broadcast_ones,key_mask])

    return mask

def gelu(x):
    pi = 3.141592653589793
    cdf = 0.5 * (1.0 + tf.tanh(
        (tf.math.sqrt(2 / pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def reshape_2D_tensor_to_3D_matrix(output_tensor, batch_size, sequence_length, hidden_size):
    return tf.reshape(output_tensor, [batch_size, sequence_length , hidden_size])

def positional_encoding(position=Config.max_length, d_model=Config.hidden_size):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    #return seq[:, tf.newaxis, :, tf.newaxis]  # (batch_size, 1, seq_len, 1)

#这里需要输入的是2D的tokens
def create_mutual_padding_mask(query_inputs,key_inputs):

    query_mask = tf.cast(tf.math.equal(query_inputs, 0), tf.float32)
    key_mask = tf.cast(tf.math.equal(key_inputs, 0), tf.float32)

    query_mask = tf.expand_dims(query_mask,axis=2)
    key_mask = tf.expand_dims(key_mask,axis=1)

    #这里生成的mask是3D的，即 [ batch_size , query_length , key_length ]
    mask = query_mask * key_mask
    mask = tf.expand_dims(mask,axis=1)
    return mask


if __name__ == "__main__":
    embedding = tf.random.normal(shape=(2,24,1024))
    AlbertEncoderLayer()([embedding,None])
    #output_tensor = AlbertEncoderLayer()([embedding, token_mask])
    #print(output_tensor)