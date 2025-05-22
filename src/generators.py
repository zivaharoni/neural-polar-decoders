import tensorflow as tf
from src.channels import BPSK, AWGN, Ising


def info_bits_generator(batch, info_bits_num):
    while True:
        info_bits = tf.random.uniform(shape=(batch, info_bits_num), minval=0, maxval=2, dtype=tf.int32)
        yield info_bits

def iid_awgn_generator(batch, N, snr_db=0.0, p=0.5):
    bpsk_modulator = BPSK()
    awgn_channel = AWGN(snr_db)
    while True:
        x = tf.cast(tf.random.uniform((batch, N, 1)) < p, tf.int32)
        c = bpsk_modulator(x)
        y = awgn_channel(c)
        yield x, y

def iid_ising_generator(batch, N, p=0.5):
    channel = Ising()
    while True:
        x = tf.cast(tf.random.uniform((batch, N, 1)) < p, tf.int32)
        y = channel(x)
        yield x, y

def custom_dataset(batch, N,  encoder, channel, mc_length=1000):
    # def bit_generator():
    #     while True:
    #         yield tf.random.uniform(
    #             shape=(batch, N,), minval=0, maxval=2, dtype=tf.int32
    #         )
    #
    # input_data = tf.data.Dataset.from_generator(
    #     bit_generator,
    #     output_signature=tf.TensorSpec(shape=(batch, N,), dtype=tf.int32)
    # )
    input_data = tf.data.Dataset.from_tensor_slices(
        tf.random.uniform((batch*mc_length, N, 1), minval=0, maxval=2, dtype=tf.int32)).batch(batch, drop_remainder=True)

    def apply_model(b):
        x, f, u, p_u = encoder(b)
        y = channel(x)
        return x, y

    return input_data.map(apply_model)