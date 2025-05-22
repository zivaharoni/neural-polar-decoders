import tensorflow as tf
from src.models import NeuralPolarDecoder, NeuralPolarDecoderHondaYamamoto, NeuralPolarDecoderOptimize
from keras.models import Sequential
from keras.layers import Dense, Embedding
from src.layers import NodeNN, Embedding2Prob, ConstEmbedding
from src.input_distribution import BinaryRNN

def build_neural_polar_decoder(config):
    checknode_nn = NodeNN(hidden_dim=config["hidden_dim"],
                          embedding_dim=config['embedding_dim'],
                          layers=config["layers"],
                          activation=config["activation"],
                          use_bias=config["use_bias"],
                          dropout=config["dropout"])
    bitnode_nn = NodeNN(hidden_dim=config["hidden_dim"],
                        embedding_dim=config['embedding_dim'],
                        layers=config["layers"],
                        activation=config["activation"],
                        use_bias=config["use_bias"],
                        dropout=config["dropout"])
    emb2llr_nn = Embedding2Prob()
    embedding_labels_nn = Embedding(input_dim=2,
                                    output_dim=config["embedding_dim"],
                                    trainable=True)
    return checknode_nn, bitnode_nn, emb2llr_nn, embedding_labels_nn

def build_neural_polar_decoder_iid_synced(config, input_shape, load_path=None):
    embedding_nn_channel = Sequential([Dense(config["hidden_dim"], activation=config["activation"]),
                               Dense(config["embedding_dim"], activation=None)],
                              name="embedding_channel_nn")
    checknode_nn, bitnode_nn, emb2llr_nn, embedding_labels_nn = build_neural_polar_decoder(config)
    model = NeuralPolarDecoder(
                embedding_nn=embedding_nn_channel,
                checknode_nn=checknode_nn,
                bitnode_nn=bitnode_nn,
                emb2llr_nn=emb2llr_nn,
                embedding_labels_nn=embedding_labels_nn,

    )
    model.build(input_shape)

    if load_path is not None:
        try:
            model([tf.zeros(shape, dtype=tf.int32) for shape in input_shape])
            model.load_weights(load_path)
            print(f"Loaded weights from {load_path}")
        except Exception as e:
                print(f"Model path {load_path} does not exist. Skipping loading weights.")
    return model

def build_neural_polar_decoder_hy_synced(config, input_shape, load_path=None):
    embedding_nn_const = Sequential([ConstEmbedding(config["embedding_dim"])],
                                    name="embedding_const_nn")
    embedding_nn_channel = Sequential([Dense(config["hidden_dim"], activation=config["activation"]),
                               Dense(config["embedding_dim"], activation=None)],
                              name="embedding_channel_nn")
    checknode_nn, bitnode_nn, emb2llr_nn, embedding_labels_nn = build_neural_polar_decoder(config)

    npd_const = NeuralPolarDecoder(
        embedding_nn=embedding_nn_const,
        checknode_nn=checknode_nn,
        bitnode_nn=bitnode_nn,
        emb2llr_nn=emb2llr_nn,
        embedding_labels_nn=embedding_labels_nn,
                build_metrics=False)

    npd_channel = NeuralPolarDecoder(
                embedding_nn=embedding_nn_channel,
                checknode_nn=checknode_nn,
                bitnode_nn=bitnode_nn,
                emb2llr_nn=emb2llr_nn,
                embedding_labels_nn=embedding_labels_nn,
                build_metrics=False)


    model = NeuralPolarDecoderHondaYamamoto(npd_const, npd_channel)
    model.build(input_shape)

    if load_path is not None:
        try:
            model([tf.zeros(shape, dtype=tf.int32) for shape in input_shape])
            model.load_weights(load_path)
            print(f"Loaded weights from {load_path}")
        except Exception as e:
            print(e)
            print(f"Model path {load_path} does not exist. Skipping loading weights.")
    return model

def build_neural_polar_decoder_hy_synced_optimize(config, input_shape, channel, load_path=None):
    embedding_nn_const = Sequential([ConstEmbedding(config["embedding_dim"])],
                              name="embedding_const_nn")
    embedding_nn_channel = Sequential([Dense(config["hidden_dim"], activation=config["activation"]),
                               Dense(config["embedding_dim"], activation=None)],
                              name="embedding_channel_nn")
    checknode_nn, bitnode_nn, emb2llr_nn, embedding_labels_nn = build_neural_polar_decoder(config)

    npd_channel = NeuralPolarDecoder(
                embedding_nn=embedding_nn_channel,
                checknode_nn=checknode_nn,
                bitnode_nn=bitnode_nn,
                emb2llr_nn=emb2llr_nn,
                embedding_labels_nn=embedding_labels_nn,
                build_metrics=False)

    npd_const = NeuralPolarDecoder(
        embedding_nn=embedding_nn_const,
        checknode_nn=checknode_nn,
        bitnode_nn=bitnode_nn,
        emb2llr_nn=emb2llr_nn,
        embedding_labels_nn=embedding_labels_nn,
                build_metrics=False)

    input_distribution = BinaryRNN(config["hidden_dim"],)

    model = NeuralPolarDecoderOptimize(npd_const, npd_channel, input_distribution, channel)
    model.build(input_shape)

    if load_path is not None:
        try:
            model.load_weights(load_path, by_name=True)
            print(f"Loaded weights from {load_path}")
        except Exception as e:
            print(e)
            print(f"Model path {load_path} does not exist. Skipping loading weights.")
    return model