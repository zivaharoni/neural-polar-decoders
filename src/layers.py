import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, Dense, Dropout
from keras.initializers import GlorotNormal


class NodeNN(Model):
    def __init__(self, hidden_dim, embedding_dim, layers, activation='elu',
                 use_bias=True, dropout=0., name='node_nn', **kwargs):
        super(NodeNN, self).__init__(name=name, **kwargs)

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = layers
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout

        self._layers = [Dense(hidden_dim, activation=activation, use_bias=use_bias,
                              kernel_initializer=GlorotNormal(), name=f"{name}-layer{i}")
                        for i in range(layers)] + \
                       [Dense(embedding_dim, activation=None, use_bias=use_bias,
                              kernel_initializer=GlorotNormal(), name=f"{name}-layer{layers}")]

        self.dropout = Dropout(dropout)
        self.layer_norm1 = RMSNorm()
        self.layer_norm2 = RMSNorm()


    def call(self, inputs, training=None, *args):
        e = inputs
        for layer in self._layers:
            e = layer(e, training=training)
            e = self.dropout(e, training=training)
        return e

    def get_config(self):
        config = super(NodeNN, self).get_config()
        config.update({
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "layers": self.num_layers,
            "activation": self.activation,
            "use_bias": self.use_bias,
            "dropout": self.dropout_rate,
            "name": self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Embedding2Prob(Model):
    def __init__(self, q=2, activation='softmax', use_bias=True, name='emb2llr_nnops'):
        super(Embedding2Prob, self).__init__(name=name)
        assert activation=='softmax' or activation is None, \
            f"invalid activation type for embedding to prob layer: {activation}"
        self.layer = Dense(q, use_bias=use_bias, activation=activation,
                           kernel_initializer=GlorotNormal(),)

    def call(self, inputs, training=None, **kwargs):
        e = inputs
        e = self.layer.__call__(e, training=training)
        return e


class ConstEmbedding(Layer):
    def __init__(self, embedding_dim, **kwargs):
        super(ConstEmbedding, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.emb = self.add_weight(
            shape=(self.embedding_dim,),
            initializer=GlorotNormal(),
            trainable=True,
            name="const_embedding"
        )

    def call(self, inputs):
        B, N = tf.shape(inputs)[0], tf.shape(inputs)[1]
        emb_tiled = tf.reshape(self.emb, (1, 1, self.embedding_dim))  # shape (1, 1, d)
        return tf.tile(emb_tiled, (B, N, 1))  # shape (B, N, d)

    def get_config(self):
        config = super().get_config()
        config.update({"embedding_dim": self.embedding_dim})
        return config


class RMSNorm(Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super(RMSNorm, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.gamma = None

    def build(self, input_shape):
        # Learnable scale parameter γ (gamma), one per feature
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),  # Scale parameter per feature
            initializer="ones",
            trainable=True,
            name="gamma"
        )
        super(RMSNorm, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Compute RMS(x) along the last dimension
        rms = tf.sqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.epsilon)
        return (inputs / rms) * self.gamma  # Normalize and scale

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config
