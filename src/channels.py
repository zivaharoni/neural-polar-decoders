import tensorflow as tf
from keras.layers import Layer


class BPSK(Layer):
    def __init__(self, **kwargs):
        super(BPSK, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        return tf.cast(inputs, tf.float32) * 2 - 1


class AWGN(Layer):
    def __init__(self, snr_db, **kwargs):
        super(AWGN, self).__init__(**kwargs)
        snr = 10 ** (snr_db / 10.0)
        self.noise_std = tf.math.rsqrt(tf.cast(snr, tf.float32))

    def call(self, inputs, *args, **kwargs):
        noise = tf.random.normal(tf.shape(inputs), stddev=self.noise_std)
        return inputs + noise


class Ising(Layer):
    def __init__(self, **kwargs):
        super(Ising, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        x = inputs
        s0 = tf.random.uniform(shape=(x.shape[0], 1, 1), minval=0, maxval=2, dtype=tf.int32)
        s = tf.concat([s0, x[:, :-1, :]], axis=1)
        noise = tf.random.uniform(shape=tf.shape(x), minval=0, maxval=2, dtype=tf.int32)
        y = tf.where(tf.equal(noise, 0), x, s)
        return y