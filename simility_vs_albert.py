from fintunning_config import Config
import albert_pretraintrain
import tensorflow as tf


class Simility:
    def __init__(self):
        self.encoder_model = albert_pretraintrain.get_albert_pretrain_model(load_sequence_output=True)
        self.last_dense = tf.keras.layers.Dense(2,activation=tf.nn.tanh)

    def __call__(self,inputs):
        token_input,token_tyoe_input = inputs
        albert_sequence_output = self.encoder_model([token_input,token_tyoe_input])


        albert_pooled_output = tf.keras.layers.GlobalMaxPool1D()(albert_sequence_output)
        albert_pooled_output = self.last_dense(albert_pooled_output)

        simility_output = albert_pooled_output
        return simility_output

if __name__ == "__main__":
    token_input = tf.keras.Input(shape=(Config.max_length,))
    token_type_input = tf.keras.Input(shape=(Config.max_length,))
    simility_output = Simility()([token_input,token_type_input])
    model = tf.keras.Model((token_input,token_type_input),(simility_output))
    print(model.summary())







