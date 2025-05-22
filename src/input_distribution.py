import tensorflow as tf
import keras
class BinaryRNN(keras.Model):
    def __init__(self, units):
        super(BinaryRNN, self).__init__()
        self.lstm = keras.layers.LSTM(units, return_sequences=True, return_state=True)
        self.dense = keras.layers.Dense(1, activation=None, use_bias=True)

    def call(self, inputs, states=None, return_state=False, training=None):
        if states is None:
            outputs, state_h, state_c = self.lstm(inputs)
        else:
            outputs, state_h, state_c = self.lstm(inputs, initial_state=states)
        outputs = self.dense(outputs)
        if return_state:
            return outputs, (state_h, state_c)
        else:
            return outputs

    def generate(self, length, batch_size=1):
        # Start with an initial random binary input for the entire batch
        states = None
        generated_sequences = tf.TensorArray(dtype=tf.float32, size=length)
        generated_llrs = tf.TensorArray(dtype=tf.float32, size=length)
        input_seq = tf.cast(tf.random.uniform((batch_size, 1, 1), minval=0, maxval=2, dtype=tf.int32), tf.float32)
        for t in range(length):
            output, states = self.call(input_seq, states=states, return_state=True)
            # Sample a binary event based on the predicted probability for each batch element
            sampled_bits = tf.cast(tf.random.uniform((batch_size, 1, 1)) < tf.math.sigmoid(output), tf.float32)
            generated_sequences = generated_sequences.write(t, tf.squeeze(sampled_bits, axis=-1))
            generated_llrs = generated_llrs.write(t, tf.squeeze(output, axis=-1))
            input_seq = tf.cast(sampled_bits, dtype=tf.float32)  # Prepare next input
        binary_sequences = tf.transpose(generated_sequences.stack(), [1, 0, 2])
        llrs = tf.transpose(generated_llrs.stack(), [1, 0, 2])
        return tf.cast(binary_sequences, tf.int32), llrs