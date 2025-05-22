import tensorflow as tf
from keras.layers import Layer


class SplitEvenOdd(Layer):
    def __init__(self, axis, name='SplitEvenOdd'):
        super(SplitEvenOdd, self).__init__(name=name)
        self.axis = axis

    def call(self, inputs, *args, **kwargs):
        shape = tf.shape(inputs)
        y_ = tf.reshape(inputs,
                        tf.concat((shape[0:self.axis], [shape[self.axis] // 2, 2], shape[self.axis+1:]), 0))
        start = tf.zeros(tf.rank(inputs)+1, tf.int32)
        y_odd = tf.squeeze(tf.slice(y_,
                                    start,
                                    tf.concat((shape[0:self.axis], [shape[self.axis] // 2, 1], shape[self.axis+1:]), 0)
                                    ), axis=self.axis+1)
        start = tf.tensor_scatter_nd_update(start, [[self.axis+1]], [1])
        y_even = tf.squeeze(tf.slice(y_,
                                     start,
                                     tf.concat((shape[0:self.axis], [shape[self.axis] // 2, 1], shape[self.axis+1:]), 0)
                                     ), axis=self.axis+1)
        return y_odd, y_even


class Interleave(Layer):
    def __init__(self, axis=1, transpose=False, name='interleave'):
        super(Interleave, self).__init__(name=name)
        self.axis = axis
        if transpose:
            self.transpose = [-1, 2]
        else:
            self.transpose = [2, -1]

    def call(self, inputs, *args, **kwargs):
        shape = tf.shape(inputs)
        N = shape[self.axis]
        perm = tf.reshape(tf.transpose(tf.reshape(tf.range(N), self.transpose)), [-1])
        out = tf.gather(inputs, perm, axis=self.axis)
        return out


class HardDecSoftmaxLayer(Layer):
    def __init__(self, **kwargs):
        super(HardDecSoftmaxLayer, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        return tf.expand_dims(tf.cast(tf.argmax(inputs, axis=-1), inputs.dtype), axis=-1)

