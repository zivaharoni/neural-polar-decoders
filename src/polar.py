import keras
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.metrics import Mean
from src.utils import MeanTensor
from src.sc_ops import SplitEvenOdd, Interleave, HardDecSoftmaxLayer
keras.backend.set_floatx('float32')
dtype = keras.backend.floatx()
class PolarEncoder(Model):
    def __init__(self, sorted_reliabilities, info_bits_num):
        super(PolarEncoder, self).__init__()
        self.split_even_odd = SplitEvenOdd(axis=1)
        self.interleave = Interleave(axis=1)

        self.N = len(sorted_reliabilities)
        self.sorted_reliabilities = sorted_reliabilities
        self.info_set = tf.sort(tf.cast(sorted_reliabilities[:info_bits_num], tf.int32))
        self.frozen_set = tf.sort(tf.cast(sorted_reliabilities[info_bits_num:], tf.int32))

    def call(self, inputs, **kwargs):
        info_bits = inputs
        batch = info_bits.shape[0]
        info_bits_num = self.info_set.shape[0]

        u_random = tf.random.uniform(shape=(batch, self.N), minval=0, maxval=2, dtype=tf.int32)
        if info_bits_num > 0:

            batch_range = tf.range(batch, dtype=tf.int32)

            i = tf.repeat(batch_range, repeats=info_bits_num)  # shape: [batch_size * |A|]
            j = tf.tile(self.info_set, [batch])
            info_indices = tf.stack([i, j], axis=1)

            updates_info = tf.reshape(info_bits, [-1])
            u = tf.tensor_scatter_nd_update(u_random, info_indices, updates_info)
            u = u[..., None]
            updates_frozen = 2 * tf.ones(shape=(batch * info_bits_num), dtype=tf.int32)
            f = tf.tensor_scatter_nd_update(u_random, info_indices, updates_frozen)
            f = f[..., None]
        else:
            u = u_random[..., None]
            f = u_random[..., None]

        x = self.transform(u)
        r = tf.cast(tf.stack((1-u_random, u_random), axis=-1), tf.float32)
        return x, f, u, tf.ones(shape=(batch, self.N, 2))*0.5, r

    @tf.function
    def transform(self, u):
        # Initialize transformed tensor
        N = u.shape[1]
        v = tf.identity(u)
        # Iteratively perform the transformation
        num_of_splits = 1
        V = list([v])
        while N > 1:
            V_1 = list([])
            V_2 = list([])
            # split into even and odd indices with respect to the depth
            for v in V:
                # compute bits amd embeddings in next layer
                v_odd, v_even = self.split_even_odd.call(v)
                V_1.append(v_odd)
                V_2.append(v_even)

            # compute all the bits in the next stage
            V_odd = tf.concat(V_1, axis=1)
            V_even = tf.concat(V_2, axis=1)
            v_xor = tf.math.floormod(V_odd + V_even, 2)
            V_xor = tf.split(v_xor, num_or_size_splits=2 ** (num_of_splits - 1), axis=1)
            V_identity = tf.split(V_even, num_or_size_splits=2 ** (num_of_splits - 1), axis=1)
            v = tf.concat([elem for pair in zip(V_xor, V_identity) for elem in pair], axis=1)
            V_ = tf.split(v, num_or_size_splits=2 ** num_of_splits, axis=1)

            V = V_
            N //= 2
            num_of_splits += 1

        return v

    def build(self, input_shape):
        super().build(input_shape)


class SCEncoder(Model):
    def __init__(self, sorted_reliabilities, info_bits_num, decoder, threshold=0.25):
        super(SCEncoder, self).__init__()
        self.split_even_odd = SplitEvenOdd(axis=1)
        self.interleave = Interleave(axis=1)
        self.hard_decision = HardDecSoftmaxLayer()
        self.decoder = decoder
        self.N = len(sorted_reliabilities)

        self.sorted_reliabilities = sorted_reliabilities
        self.info_set = tf.sort(tf.cast(sorted_reliabilities[:info_bits_num], tf.int32))
        self.frozen_set = tf.sort(tf.cast(sorted_reliabilities[info_bits_num:], tf.int32))
        self.threshold = threshold
        print(f"SC encoder: info bits num: {info_bits_num}, threshold: {self.threshold}" )

    def call(self, inputs, **kwargs):
        info_bits = inputs
        batch = info_bits.shape[0]
        info_bits_num = self.info_set.shape[0]
        u_ = 2 * tf.ones(shape=(batch, self.N), dtype=tf.int32)
        e = self.decoder.embedding_observations_nn(u_[..., None])
        r = tf.random.uniform(shape=(batch, self.N), dtype=tf.float32)
        if info_bits_num > 0:
            info_bits = info_bits[:, :info_bits_num]
            batch_range = tf.range(batch, dtype=tf.int32)

            i = tf.repeat(batch_range, repeats=info_bits_num)  #
            j = tf.tile(self.info_set, [batch])
            info_indices = tf.stack([i, j], axis=1)
            updates_info = tf.reshape(info_bits, [-1])
            f_enc = 2 * tf.ones(shape=(batch, self.N, 1), dtype=tf.int32)
            f_enc = tf.tensor_scatter_nd_update(f_enc[...,0], info_indices, updates_info)[...,None]

            u, x, p_u = self.encode(e, f_enc, r, self.N)

            updates_info = 2 * tf.ones(shape=(batch * info_bits_num), dtype=tf.int32)
            f = tf.tensor_scatter_nd_update(tf.ones_like(f_enc[...,0]), info_indices, updates_info)[...,None]

        else:
            u_ = u_[..., None]

            u, x, p_u = self.encode(e, u_, r, self.N)
            f = tf.identity(u)

        return x, f, u, p_u, r

    @tf.function
    def encode(self, e, f, r, N, *args):
        if N == 1:
            p_u = self.decoder.emb2llr_nn(e, training=False)
            hard_decision = tf.cast(r > p_u[..., 0], tf.int32)[..., None]

            x = tf.where(tf.logical_or(tf.equal(f, 2), self.frozen_cond(p_u)), hard_decision, f)

            u = tf.identity(x)

            return u, x, p_u

        e_odd, e_even = self.split_even_odd.call(e)
        f_halves = tf.split(f, num_or_size_splits=2, axis=1)
        f_left, f_right = f_halves
        r_halves = tf.split(r, num_or_size_splits=2, axis=1)
        r_left, r_right = r_halves


        # Compute soft mapping back one stage
        u1est = self.decoder.checknode_nn.call(tf.concat((e_odd, e_even), axis=-1), training=False)
        # u1est = self.layer_norms[layer_norm_pointer](u1est, training=False)
        # R_N^T maps u1est to top polar code
        uhat1, u1hardprev, p_u_left = self.encode(u1est, f_left, r_left, N // 2)
        u_emb = self.decoder.embedding_labels_nn(tf.squeeze(u1hardprev, axis=-1))

        # Using u1est and x1hard, we can estimate u2
        u2est = self.decoder.bitnode_nn.call(tf.concat((e_odd, e_even, u_emb), axis=-1), training=False)
        # u2est = self.layer_norms[layer_norm_pointer](u2est, training=False)

        # R_N^T maps u2est to bottom polar code
        uhat2, u2hardprev, p_u_right = self.encode(u2est, f_right, r_right, N // 2)

        u = tf.concat([uhat1, uhat2], axis=1)
        p_u = tf.concat([p_u_left, p_u_right], axis=1)

        v_xor = tf.math.floormod(u1hardprev + u2hardprev, 2)
        v_identity = tf.identity(u2hardprev)
        x = self.interleave.call(tf.concat((v_xor, v_identity), axis=1))
        return u, x, p_u

    def build(self, input_shape):
        super().build(input_shape)

    def frozen_cond(self, p_u):
        return tf.greater(tf.abs(p_u[..., 0:1]- 0.5), self.threshold)
        super().build(input_shape)


class SCDecoder(Model):
    def __init__(self, decoder):
        super(SCDecoder, self).__init__()

        self.decoder = decoder
        self.hard_decision = HardDecSoftmaxLayer()
        self.interleave = Interleave(axis=1)
        self.split_even_odd = SplitEvenOdd(axis=1)

    def call(self, inputs, **kwargs):
        y, f = inputs
        e = self.decoder.embedding_observations_nn(y, training=False)

        uhat, xhat, llr_u1 = self.decode(e, f, f.shape[1])

        return uhat, llr_u1

    @tf.function
    def decode(self, e, f, N, *args):
        if N == 1:
            p_uy = self.decoder.emb2llr_nn(e, training=False)
            hard_decision = tf.cast(self.hard_decision.call(p_uy), dtype=tf.int32)
            u = tf.where(tf.equal(f, 2), hard_decision, f)
            x = tf.identity(u)

            return u, x, p_uy

        e_odd, e_even = self.split_even_odd.call(e)
        f_halves = tf.split(f, num_or_size_splits=2, axis=1)
        f_left, f_right = f_halves

        # Compute soft mapping back one stage
        u1est = self.decoder.checknode_nn.call(tf.concat((e_odd, e_even), axis=-1), training=False)
        # u1est = self.layer_norms[layer_norm_pointer](u1est, training=False)
        # R_N^T maps u1est to top polar code
        uhat1, u1hardprev, p_uy_left = self.decode(u1est, f_left, N // 2)
        u_emb = self.decoder.embedding_labels_nn(tf.squeeze(u1hardprev, axis=-1))

        # Using u1est and x1hard, we can estimate u2
        u2est = self.decoder.bitnode_nn.call(tf.concat((e_odd, e_even, u_emb), axis=-1), training=False)
        # u2est = self.layer_norms[layer_norm_pointer](u2est, training=False)

        # R_N^T maps u2est to bottom polar code
        uhat2, u2hardprev, p_uy_right = self.decode(u2est, f_right, N // 2)

        u = tf.concat([uhat1, uhat2], axis=1)
        p_uy = tf.concat([p_uy_left, p_uy_right], axis=1)

        v_xor = tf.math.floormod(u1hardprev + u2hardprev, 2)
        v_identity = tf.identity(u2hardprev)
        x = self.interleave.call(tf.concat((v_xor, v_identity), axis=1))
        return u, x, p_uy

    def build(self, input_shape):
        super().build(input_shape)


class SCLDecoder(SCDecoder):
    def __init__(self, decoder, list_num=4):
        super(SCLDecoder, self).__init__(decoder)

        self.interleave = Interleave(axis=2)
        self.split_even_odd = SplitEvenOdd(axis=2)
        self.list_num = list_num
        self.eps = 1e-6

    def call(self, inputs, **kwargs):
        y, f = inputs
        f = tf.tile(tf.expand_dims(f, 1), [1, self.list_num, 1, 1])

        r = tf.random.uniform(shape=(y.shape[0], y.shape[1], 1), dtype=tf.float32)
        r = tf.tile(tf.expand_dims(r, 1), [1, self.list_num, 1, 1])

        e = self.decoder.embedding_observations_nn(y, training=False)
        e = tf.expand_dims(e, 1)
        repmat = tf.tensor_scatter_nd_update(tensor=tf.ones_like(tf.shape(e)),
                                             indices=tf.constant([[1]]),
                                             updates=tf.constant([self.list_num]))
        e = tf.tile(e, repmat)

        maxllr = 10 ** 8
        pm = tf.concat([tf.zeros([1]), tf.ones([self.list_num - 1]) * float(maxllr)], 0)
        pm = tf.tile(tf.expand_dims(pm, 0), [f.shape[0], 1])
        uhat_list, xhat, llr_uy, pm, new_order = self.decode(e, f, pm,
                                                                  f.shape[2], r, sample=True)

        uhat = tf.gather(uhat_list, tf.argmin(pm, axis=1), axis=1, batch_dims=1)


        return uhat, llr_uy

    @tf.function
    def decode(self, e, f, pm, N, r, sample=True, *args):

        nL = e.shape[1]
        if N == 1:
            frozen = f
            dm = self.decoder.emb2llr_nn(e, training=False)
            # Ensure probabilities are clipped to avoid log(0) or division by zero
            p1_safe = tf.clip_by_value(dm[..., 1], self.eps, 1 - self.eps)
            p0_safe = 1.0 - p1_safe  # tf.clip_by_value(dm[..., 0], epsilon, 1 - epsilon)

            # Compute the log-likelihood ratio
            llr = tf.math.log(p1_safe) - tf.math.log(p0_safe)
            llr = tf.expand_dims(llr, axis=-1)
            hd_ = tf.squeeze(self.hard_decision.call(dm), axis=(2, 3))
            hd_ = tf.cast(hd_, dtype=tf.int32)
            # print(hd_.shape)

            hd = tf.concat((hd_, 1 - hd_), axis=1)
            # print(hd.shape)

            pm_dup = tf.concat((pm, pm + tf.abs(tf.squeeze(llr, axis=(2, 3)))), -1)
            pm_prune, prune_idx_ = tf.math.top_k(-pm_dup, k=nL, sorted=True)
            pm_prune = -pm_prune
            prune_idx = tf.sort(prune_idx_, axis=1)
            idx = tf.argsort(prune_idx_, axis=1)
            pm_prune = tf.gather(pm_prune, idx, axis=1, batch_dims=1)
            u_survived = tf.gather(hd, prune_idx, axis=1, batch_dims=1)[:, :, tf.newaxis, tf.newaxis]
            # print(u_survived.shape)
            is_frozen = tf.not_equal(f, 2)
            x = tf.where(is_frozen, frozen, u_survived)
            # print(is_frozen.shape, frozen.shape)

            pm_ = tf.where(tf.squeeze(is_frozen, axis=(2, 3)),
                           pm + tf.abs(tf.squeeze(llr, axis=(2, 3))) *
                           tf.cast(tf.squeeze(tf.not_equal(tf.expand_dims(tf.expand_dims(hd_, -1), -1), frozen),
                                              axis=(2, 3)), tf.float32),
                           pm_prune)
            new_order = tf.tile(tf.expand_dims(tf.range(nL), 0), [e.shape[0], 1]) \
                if f[0, 0, 0, 0] != 2 else (prune_idx % nL)
            return x, tf.identity(x), llr, pm_, new_order

        f_halves = tf.split(f, num_or_size_splits=2, axis=2)
        f_left, f_right = f_halves
        r_halves = tf.split(r, num_or_size_splits=2, axis=2)
        r_left, r_right = r_halves

        ey_odd, ey_even = self.split_even_odd.call(e)

        # Compute soft mapping back one stage
        ey1est = self.decoder.checknode_nn.call(tf.concat((ey_odd, ey_even), axis=-1))
        shape = ey1est.shape
        # ey1est = self.layer_norms[layer_norm_pointer](tf.reshape(ey1est, [-1, shape[-2], shape[-1]]), training=False)
        ey1est = tf.reshape(ey1est, shape)
        # R_N^T maps u1est to top polar code
        uhat1, u1hardprev, llr_uy_left, pm, new_order = self.decode(ey1est, f_left, pm,
                                                                         N // 2, r_left, sample=sample)

        # Using u1est and x1hard, we can estimate u2

        ey_odd = tf.gather(ey_odd, new_order, axis=1, batch_dims=1)
        ey_even = tf.gather(ey_even, new_order, axis=1, batch_dims=1)

        # Using u1est and x1hard, we can estimate u2
        u_emb = tf.squeeze(self.decoder.embedding_labels_nn(u1hardprev), axis=-2)
        ey2est = self.decoder.bitnode_nn.call(tf.concat((ey_odd, ey_even, u_emb), axis=-1))


        # Using u1est and x1hard, we can estimate u2
        uhat2, u2hardprev, llr_uy_right, pm, new_order2 = self.decode(ey2est, f_right, pm,
                                                                           N // 2, r_right, sample=sample)
        uhat1 = tf.gather(uhat1, new_order2, axis=1, batch_dims=1)
        llr_uy_left = tf.gather(llr_uy_left, new_order2, axis=1, batch_dims=1)
        u1hardprev = tf.gather(u1hardprev, new_order2, axis=1, batch_dims=1)
        new_order = tf.gather(new_order, new_order2, axis=1, batch_dims=1)
        u = tf.concat([uhat1, uhat2], axis=2)
        llr_uy = tf.concat([llr_uy_left, llr_uy_right], axis=2)
        v_xor = tf.math.floormod(u1hardprev + u2hardprev, 2)
        v_identity = tf.identity(u2hardprev)
        x = self.interleave.call(tf.concat((v_xor, v_identity), axis=2))
        return u, x, llr_uy, pm, new_order

    def build(self, input_shape):
        super().build(input_shape)


class SCDecoderHY(Model):
    def __init__(self, encoder, decoder):
        super(SCDecoderHY, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.hard_decision = HardDecSoftmaxLayer()
        self.interleave = Interleave(axis=1)
        self.split_even_odd = SplitEvenOdd(axis=1)

    def call(self, inputs, **kwargs):
        y, f, r  = inputs
        e_co = self.encoder.embedding_observations_nn(2*tf.ones_like(y), training=False)
        e_ch = self.decoder.embedding_observations_nn(y, training=False)

        uhat, xhat, llr_u1 = self.decode(e_co, e_ch, f, r, f.shape[1])

        return uhat, llr_u1

    @tf.function
    def decode(self, e_co, e_ch, f, r, N, *args):
        if N == 1:
            p_u = self.encoder.emb2llr_nn(e_co, training=False)
            p_uy = self.decoder.emb2llr_nn(e_ch, training=False)
            hard_decision_u = tf.cast(r > p_u[..., 0], tf.int32)[..., None]
            # hard_decision_u = f
            # arg = tf.math.log(p_uy+1e-10) - tf.math.log(p_u+1e-10)
            # hard_decision_uy = tf.cast(tf.argmax(arg, axis=-1)[...,None], dtype=tf.int32)
            hard_decision_uy = tf.cast(tf.argmax(p_uy, axis=-1)[...,None], dtype=tf.int32)
            u = tf.where(tf.equal(f, 2), hard_decision_uy, hard_decision_u)
            x = tf.identity(u)
            return u, x, p_uy

        e_co_odd, e_co_even = self.split_even_odd.call(e_co)
        e_ch_odd, e_ch_even = self.split_even_odd.call(e_ch)
        f_halves = tf.split(f, num_or_size_splits=2, axis=1)
        f_left, f_right = f_halves
        r_halves = tf.split(r, num_or_size_splits=2, axis=1)
        r_left, r_right = r_halves

        # Compute soft mapping back one stage
        u1est_co = self.encoder.checknode_nn.call(tf.concat((e_co_odd, e_co_even), axis=-1), training=False)
        u1est_ch = self.decoder.checknode_nn.call(tf.concat((e_ch_odd, e_ch_even), axis=-1), training=False)

        # R_N^T maps u1est to top polar code
        uhat1, u1hardprev, p_uy_left = self.decode(u1est_co, u1est_ch, f_left, r_left, N // 2)
        u_emb = self.encoder.embedding_labels_nn(tf.squeeze(u1hardprev, axis=-1))

        # Using u1est and x1hard, we can estimate u2
        u2est_co  = self.encoder.bitnode_nn.call(tf.concat((e_co_odd, e_co_even, u_emb), axis=-1), training=False)
        u2est_ch  = self.decoder.bitnode_nn.call(tf.concat((e_ch_odd, e_ch_even, u_emb), axis=-1), training=False)

        # R_N^T maps u2est to bottom polar code
        uhat2, u2hardprev, p_uy_right = self.decode(u2est_co, u2est_ch, f_right, r_right, N // 2)

        u = tf.concat([uhat1, uhat2], axis=1)
        p_uy = tf.concat([p_uy_left, p_uy_right], axis=1)

        v_xor = tf.math.floormod(u1hardprev + u2hardprev, 2)
        v_identity = tf.identity(u2hardprev)
        x = self.interleave.call(tf.concat((v_xor, v_identity), axis=1))
        return u, x, p_uy

    def build(self, input_shape):
        super().build(input_shape)


class SCLDecoderHY(SCDecoderHY):
    def __init__(self, encoder, decoder, list_num=4, threshold=0.25):
        super(SCLDecoderHY, self).__init__(encoder, decoder)

        self.interleave = Interleave(axis=2)
        self.split_even_odd = SplitEvenOdd(axis=2)
        self.list_num = list_num
        self.eps = 1e-6
        self.threshold = threshold

        print(f"SCL decoder: list num: {self.list_num}, threshold: {self.threshold}" )

    def call(self, inputs, **kwargs):
        y, f, r  = inputs
        f = tf.tile(tf.expand_dims(f, 1), [1, self.list_num, 1, 1])

        r = tf.tile(tf.expand_dims(r[...,None], 1), [1, self.list_num, 1, 1])

        e_co = self.encoder.embedding_observations_nn(2*tf.ones_like(y), training=False)
        e_co = tf.expand_dims(e_co, 1)
        e_ch = self.decoder.embedding_observations_nn(y, training=False)
        e_ch = tf.expand_dims(e_ch, 1)

        repmat = tf.tensor_scatter_nd_update(tensor=tf.ones_like(tf.shape(e_ch)),
                                             indices=tf.constant([[1]]),
                                             updates=tf.constant([self.list_num]))
        e_co = tf.tile(e_co, repmat)
        e_ch = tf.tile(e_ch, repmat)

        maxllr = 10 ** 8
        pm = tf.concat([tf.zeros([1]), tf.ones([self.list_num - 1]) * float(maxllr)], 0)
        pm = tf.tile(tf.expand_dims(pm, 0), [f.shape[0], 1])
        uhat_list, xhat, p_uy_list, p_u_list, pm, new_order = self.decode(e_co, e_ch, f, pm,
                                                                  f.shape[2], r, sample=True)

        uhat = tf.gather(uhat_list, tf.argmin(pm, axis=1), axis=1, batch_dims=1)
        p_uy = tf.gather(p_uy_list, tf.argmin(pm, axis=1), axis=1, batch_dims=1)
        p_u = tf.gather(p_u_list, tf.argmin(pm, axis=1), axis=1, batch_dims=1)


        return uhat, p_uy, p_u

    @tf.function
    def decode(self, e_co, e_ch, f, pm, N, r, sample=True, *args):

        nL = e_ch.shape[1]
        if N == 1:
            p_u = self.encoder.emb2llr_nn(e_co, training=False)
            frozen = tf.cast(r > p_u[..., 0:1], tf.int32)

            dm = self.decoder.emb2llr_nn(e_ch, training=False)
            # Ensure probabilities are clipped to avoid log(0) or division by zero
            p1_safe = tf.clip_by_value(dm[..., 1], self.eps, 1 - self.eps)
            p0_safe = 1.0 - p1_safe

            # Compute the log-likelihood ratio
            llr = tf.math.log(p1_safe) - tf.math.log(p0_safe)
            llr = tf.expand_dims(llr, axis=-1)
            hd_ = tf.squeeze(self.hard_decision.call(dm), axis=(2, 3))
            hd_ = tf.cast(hd_, dtype=tf.int32)

            hd = tf.concat((hd_, 1 - hd_), axis=1)

            pm_dup = tf.concat((pm, pm + tf.abs(tf.squeeze(llr, axis=(2, 3)))), -1)
            pm_prune, prune_idx_ = tf.math.top_k(-pm_dup, k=nL, sorted=True)
            pm_prune = -pm_prune
            prune_idx = tf.sort(prune_idx_, axis=1)
            idx = tf.argsort(prune_idx_, axis=1)
            pm_prune = tf.gather(pm_prune, idx, axis=1, batch_dims=1)
            u_survived = tf.gather(hd, prune_idx, axis=1, batch_dims=1)[:, :, tf.newaxis, tf.newaxis]

            is_frozen = tf.not_equal(f, 2)   
            
            x = tf.where(is_frozen, frozen, u_survived)


            pm_ = tf.where(tf.squeeze(is_frozen, axis=(2, 3)),
                           pm + tf.abs(tf.squeeze(llr, axis=(2, 3))) *
                           tf.cast(tf.squeeze(tf.not_equal(tf.expand_dims(tf.expand_dims(hd_, -1), -1), frozen),
                                              axis=(2, 3)), tf.float32),
                           pm_prune)
            new_order = tf.tile(tf.expand_dims(tf.range(nL), 0), [e_ch.shape[0], 1]) \
                if f[0, 0, 0, 0] != 2 else (prune_idx % nL)
            return x, tf.identity(x),  dm, p_u, pm_, new_order

        f_halves = tf.split(f, num_or_size_splits=2, axis=2)
        f_left, f_right = f_halves
        r_halves = tf.split(r, num_or_size_splits=2, axis=2)
        r_left, r_right = r_halves

        e_co_odd, e_co_even = self.split_even_odd.call(e_co)
        e_ch_odd, e_ch_even = self.split_even_odd.call(e_ch)

        # Compute soft mapping back one stage
        u1est_co = self.encoder.checknode_nn.call(tf.concat((e_co_odd, e_co_even), axis=-1), training=False)
        u1est_ch = self.decoder.checknode_nn.call(tf.concat((e_ch_odd, e_ch_even), axis=-1), training=False)

        shape = u1est_ch.shape
        # ey1est = self.layer_norms[layer_norm_pointer](tf.reshape(ey1est, [-1, shape[-2], shape[-1]]), training=False)
        # R_N^T maps u1est to top polar code
        uhat1, u1hardprev,  p_uy_left, p_u_left, pm, new_order = self.decode(u1est_co, u1est_ch, f_left, pm,
                                                                         N // 2, r_left, sample=sample)



        # Using u1est and x1hard, we can estimate u2

        e_co_odd = tf.gather(e_co_odd, new_order, axis=1, batch_dims=1)
        e_co_even = tf.gather(e_co_even, new_order, axis=1, batch_dims=1)
        e_ch_odd = tf.gather(e_ch_odd, new_order, axis=1, batch_dims=1)
        e_ch_even = tf.gather(e_ch_even, new_order, axis=1, batch_dims=1)

        # Using u1est and x1hard, we can estimate u2
        u_emb_co = tf.squeeze(self.encoder.embedding_labels_nn(u1hardprev), axis=-2)
        u_emb_ch = tf.squeeze(self.decoder.embedding_labels_nn(u1hardprev), axis=-2)


        ey2est_co = self.encoder.bitnode_nn.call(tf.concat((e_co_odd, e_co_even, u_emb_co), axis=-1))
        ey2est_ch = self.decoder.bitnode_nn.call(tf.concat((e_ch_odd, e_ch_even, u_emb_ch), axis=-1))


        # Using u1est and x1hard, we can estimate u2
        uhat2, u2hardprev, p_uy_right, p_u_right, pm, new_order2 = self.decode(ey2est_co, ey2est_ch, f_right, pm,
                                                                           N // 2, r_right, sample=sample)
        
        uhat1 = tf.gather(uhat1, new_order2, axis=1, batch_dims=1)
        p_u_left = tf.gather(p_u_left, new_order2, axis=1, batch_dims=1)
        p_uy_left = tf.gather(p_uy_left, new_order2, axis=1, batch_dims=1)
        u1hardprev = tf.gather(u1hardprev, new_order2, axis=1, batch_dims=1)
        new_order = tf.gather(new_order, new_order2, axis=1, batch_dims=1)

        u = tf.concat([uhat1, uhat2], axis=2)
        p_uy = tf.concat([p_uy_left, p_uy_right], axis=2)
        p_u = tf.concat([p_u_left, p_u_right], axis=2)
        v_xor = tf.math.floormod(u1hardprev + u2hardprev, 2)
        v_identity = tf.identity(u2hardprev)
        x = self.interleave.call(tf.concat((v_xor, v_identity), axis=2))
        return u, x, p_uy, p_u, pm, new_order

    def build(self, input_shape):
        super().build(input_shape)

    def frozen_cond(self, p_u):
        return tf.greater(tf.abs(p_u[..., 0:1]- 0.5), self.threshold)

class PolarCodeConstruction(Model):
    def __init__(self, encoder, modulator, channel, decoder):
        super(PolarCodeConstruction, self).__init__()
        self.encoder = encoder
        self.modulator = modulator
        self.channel = channel
        self.decoder = decoder

        self.synthetic_channel_entropy_metric_x = MeanTensor(name="synthetic_channel_entropy_x")
        self.synthetic_channel_entropy_metric_y = MeanTensor(name="synthetic_channel_entropy_y")

    def call(self, inputs, **kwargs):
        info_bits = inputs
        x, f, u, p_u, r = self.encoder(info_bits)
        c = self.modulator(x)
        y = self.channel(c)

        loss_array_y, _, _ = self.decoder.decoder((x, y), training=False)
        return tf.ones_like(loss_array_y) * tf.math.log(2.0), loss_array_y

    @tf.function
    def test_step(self, inputs):
        loss_array_x, loss_array_y = self(inputs, training=False)

        ce_x = loss_array_x[:,-1,:] / tf.math.log(2.0)
        ce_y = loss_array_y[:,-1,:] / tf.math.log(2.0)

        self.synthetic_channel_entropy_metric_x.update_state(ce_x)
        self.synthetic_channel_entropy_metric_y.update_state(ce_y)
        res = {'ce_x': tf.reduce_mean(self.synthetic_channel_entropy_metric_x.result()),
               'ce_y': tf.reduce_mean(self.synthetic_channel_entropy_metric_y.result()),
               'mi': tf.reduce_mean(self.synthetic_channel_entropy_metric_x.result() - self.synthetic_channel_entropy_metric_y.result()),
               }
        return res

    def build(self, input_shape):
        super().build(input_shape)


class PolarCodeConstructionHY(Model):
    def __init__(self, encoder, modulator, channel, decoder):
        super(PolarCodeConstructionHY, self).__init__()
        self.encoder = encoder
        self.modulator = modulator
        self.channel = channel
        self.decoder = decoder

        self.synthetic_channel_entropy_metric_x = MeanTensor(name="synthetic_channel_entropy_x")
        self.synthetic_channel_entropy_metric_y = MeanTensor(name="synthetic_channel_entropy_y")

    def call(self, inputs, **kwargs):
        info_bits = inputs
        x, f, u, p_u, r = self.encoder(info_bits)
        c = self.modulator(x)
        y = self.channel(c)

        inputs_tilde = x, 2 * tf.ones_like(y)
        loss_array_x, _, _ = self.encoder.decoder(inputs_tilde, training=False)
        loss_array_y, _, _ = self.decoder.decoder((x, y), training=False)
        return loss_array_x, loss_array_y

    @tf.function
    def test_step(self, inputs):
        loss_array_x, loss_array_y = self(inputs, training=False)

        ce_x = loss_array_x[:,-1,:] / tf.math.log(2.0)
        ce_y = loss_array_y[:,-1,:] / tf.math.log(2.0)

        self.synthetic_channel_entropy_metric_x.update_state(ce_x)
        self.synthetic_channel_entropy_metric_y.update_state(ce_y)
        res = {'ce_x': tf.reduce_mean(self.synthetic_channel_entropy_metric_x.result()),
               'ce_y': tf.reduce_mean(self.synthetic_channel_entropy_metric_y.result()),
               'mi': tf.reduce_mean(self.synthetic_channel_entropy_metric_x.result() - self.synthetic_channel_entropy_metric_y.result()),
               }
        return res

    def build(self, input_shape):
        super().build(input_shape)


class PolarCode(Model):
    def __init__(self, encoder, modulator, channel, decoder):
        super(PolarCode, self).__init__()
        self.encoder = encoder
        self.modulator = modulator
        self.channel = channel
        self.decoder = decoder

        self.ber_metric = Mean(name="ber")
        self.fer_metric = Mean(name="fer")

    def call(self, inputs, **kwargs):
        info_bits = inputs
        x, f, u, p_u, r = self.encoder(info_bits)
        c = self.modulator(x)
        y = self.channel(c)
        uhat, llr_u1 = self.decoder((y, f))
        return uhat, u

    @tf.function
    def test_step(self, inputs):
        uhat, u = self(inputs, training=False)
        errors = tf.cast(tf.not_equal(uhat, u), tf.float32)
        info_errors = tf.gather(errors, indices=self.encoder.info_set, axis=1)[..., 0]
        
        ber = info_errors
        fer = tf.cast(tf.reduce_sum(info_errors, axis=1) > 0, tf.float32)
        self.ber_metric.update_state(ber)
        self.fer_metric.update_state(fer)
        # Return a dict mapping metric names to current value
        res = {
            'ber': tf.reduce_mean(ber),
            'fer': tf.reduce_mean(fer),
               }
        return res

    def build(self, input_shape):
        super().build(input_shape)


class PolarCodeHY(PolarCode):
    def __init__(self, encoder, modulator, channel, decoder):
        super(PolarCodeHY, self).__init__(encoder, modulator, channel, decoder)
        self.code_rate_metric = Mean(name="code_rate")


    def call(self, inputs, **kwargs):
        info_bits = inputs
        x, f, u, p_u_enc, r = self.encoder(info_bits)
        c = self.modulator(x)
        y = self.channel(c)
        uhat, p_uy, p_u = self.decoder((y, f, r))
        errors = tf.cast(tf.not_equal(uhat, u), tf.float32)
        info_errors = tf.gather(errors, indices=self.encoder.info_set, axis=1)[..., 0]
        p_u_info = tf.gather(p_u, indices=self.encoder.info_set, axis=1)[..., 0]
        p_u_enc_info = tf.gather(p_u_enc, indices=self.encoder.info_set, axis=1)[..., 0]
        ragged_errors = tf.ragged.boolean_mask(info_errors, tf.logical_not(self.encoder.frozen_cond(p_u_info[..., None])[..., 0]))
        rate = tf.reduce_mean(tf.cast(ragged_errors.row_lengths(), tf.float32))
        frozen_errors = tf.not_equal(self.encoder.frozen_cond(p_u_info), 
                                                  self.encoder.frozen_cond(p_u_enc_info))
        # print(tf.reduce_mean(tf.cast(frozen_errors, tf.float32)))
        ber = tf.reduce_mean(ragged_errors)
        fer = tf.reduce_mean(tf.cast(tf.reduce_sum(ragged_errors, axis=1) > 0, tf.float32))
        return ber, fer, rate
    
    @tf.function
    def test_step(self, inputs):
        ber, fer, rate = self(inputs, training=False)


        self.ber_metric.update_state(ber)
        self.fer_metric.update_state(fer)
        self.code_rate_metric.update_state(rate)
        # Return a dict mapping metric names to current value
        res = {
            'ber': tf.reduce_mean(ber),
            'fer': tf.reduce_mean(fer),
            'rate':rate,
               }
        return res
