import keras
import tensorflow as tf
from keras.models import Model
from keras.metrics import Mean
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
        return x, f, u, tf.ones(shape=(batch, self.N, 2))*0.5

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
    def __init__(self, sorted_reliabilities, info_bits_num, decoder):
        super(SCEncoder, self).__init__()
        self.split_even_odd = SplitEvenOdd(axis=1)
        self.interleave = Interleave(axis=1)
        self.hard_decision = HardDecSoftmaxLayer()
        self.decoder = decoder
        self.N = len(sorted_reliabilities)

        self.sorted_reliabilities = sorted_reliabilities
        self.info_set = tf.sort(tf.cast(sorted_reliabilities[:info_bits_num], tf.int32))
        self.frozen_set = tf.sort(tf.cast(sorted_reliabilities[info_bits_num:], tf.int32))

    def call(self, inputs, **kwargs):
        info_bits = inputs
        batch = info_bits.shape[0]
        info_bits_num = self.info_set.shape[0]
        u_ = 2 * tf.ones(shape=(batch, self.N), dtype=tf.int32)

        if info_bits_num > 0:
            info_bits = info_bits[:, :info_bits_num]
            batch_range = tf.range(batch, dtype=tf.int32)

            i = tf.repeat(batch_range, repeats=info_bits_num)  # shape: [batch_size * |A|]
            j = tf.tile(self.info_set, [batch])
            info_indices = tf.stack([i, j], axis=1)


            updates_info = tf.reshape(info_bits, [-1])
            u_ = tf.tensor_scatter_nd_update(u_, info_indices, updates_info)
            u_ = u_[..., None]
            e = self.decoder.embedding_observations_nn(2 * tf.ones_like(u_))
            u, x, p_u = self.encode(e, u_, self.N)
            updates_frozen = 2 * tf.ones(shape=(batch * info_bits_num), dtype=tf.int32)
            f = tf.tensor_scatter_nd_update(u[...,0], info_indices, updates_frozen)
            f = f[..., None]
        else:
            u_ = u_[..., None]

            e = self.decoder.embedding_observations_nn(2*tf.ones_like(u_))
            u, x, p_u = self.encode(e, u_, self.N)
            f = u

        return x, f, u, p_u

    @tf.function
    def encode(self, e, f, N, *args):
        if N == 1:
            p_uy = self.decoder.emb2llr_nn(e, training=False)
            cdf = tf.cumsum(p_uy, axis=-1)
            rv = tf.random.uniform(shape=(e.shape[0], e.shape[1]), dtype=p_uy.dtype)
            hard_decision = tf.cast(rv > cdf[..., 0], tf.int32)[..., None]
            # tf.print(rv, cdf[..., 0])
            # tf.print(cdf.shape)
            # tf.print(rv.shape)

            # hard_decision = tf.cast(self.hard_decision.call(p_uy), dtype=tf.int32)
            x = tf.where(tf.equal(f, 2), hard_decision, f)
            u = tf.identity(x)

            return u, x, p_uy

        e_odd, e_even = self.split_even_odd.call(e)
        f_halves = tf.split(f, num_or_size_splits=2, axis=1)
        f_left, f_right = f_halves

        # Compute soft mapping back one stage
        u1est = self.decoder.checknode_nn.call(tf.concat((e_odd, e_even), axis=-1), training=False)
        # u1est = self.layer_norms[layer_norm_pointer](u1est, training=False)
        # R_N^T maps u1est to top polar code
        uhat1, u1hardprev, p_uy_left = self.encode(u1est, f_left, N // 2)
        u_emb = self.decoder.embedding_labels_nn(tf.squeeze(u1hardprev, axis=-1))

        # Using u1est and x1hard, we can estimate u2
        u2est = self.decoder.bitnode_nn.call(tf.concat((e_odd, e_even, u_emb), axis=-1), training=False)
        # u2est = self.layer_norms[layer_norm_pointer](u2est, training=False)

        # R_N^T maps u2est to bottom polar code
        uhat2, u2hardprev, p_uy_right = self.encode(u2est, f_right, N // 2)

        u = tf.concat([uhat1, uhat2], axis=1)
        p_uy = tf.concat([p_uy_left, p_uy_right], axis=1)

        v_xor = tf.math.floormod(u1hardprev + u2hardprev, 2)
        v_identity = tf.identity(u2hardprev)
        x = self.interleave.call(tf.concat((v_xor, v_identity), axis=1))
        return u, x, p_uy

    def build(self, input_shape):
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
        x, f, u, p_u = self.encoder(info_bits)
        c = self.modulator(x)
        y = self.channel(c)
        uhat, llr_u1 = self.decoder((y, f))
        # llr_u = tf.where(tf.equal(u, 1), llr_u1[:,1], llr_u1[:,0])
        # log_pu = tf.math.log(tf.math.sigmoid(llr_u) + 1e-10)
        # tf.print("dec: ", tf.reduce_mean(-log_pu))
        # tf.print("===============")
        # tf.print( tf.stack((f[0,...,0],u[0,...,0],uhat[0,...,0]), axis=0), summarize=-1)
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
            'ber': self.ber_metric.result(),
            'fer': self.fer_metric.result()
               }
        return res#

    def build(self, input_shape):
        super().build(input_shape)

    # @tf.function
    # def decode_list(self, ey, f, pm, N, r, sample=True, , *args):
    #     # layer_norm_pointer = int(np.log2(self.block_length / N))
    #     nL = ey.shape[1]
    #     if N == 1:
    #         frozen = f
    #         # print(f.shape)
    #
    #         dm = self.emb2llr.call(ey)
    #
    #         # E0nsure probabilities are clipped to avoid log(0) or division by zero
    #         p1_safe = tf.clip_by_value(dm[..., 1], self.eps, 1 - self.eps)
    #         p0_safe = 1.0 - p1_safe #tf.clip_by_value(dm[..., 0], epsilon, 1 - epsilon)
    #
    #         # Compute the log-likelihood ratio
    #         llr = tf.math.log(p1_safe) - tf.math.log(p0_safe)
    #         llr = tf.expand_dims(llr, axis=-1)
    #         hd_ = tf.squeeze(self.hard_decision.call(dm), axis=(2, 3))
    #         # print(hd_.shape)
    #
    #         hd = tf.concat((hd_, 1 - hd_), axis=1)
    #         # print(hd.shape)
    #
    #         pm_dup = tf.concat((pm, pm + tf.abs(tf.squeeze(llr, axis=(2, 3)))), -1)
    #         pm_prune, prune_idx_ = tf.math.top_k(-pm_dup, k=nL, sorted=True)
    #         pm_prune = -pm_prune
    #         prune_idx = tf.sort(prune_idx_, axis=1)
    #         idx = tf.argsort(prune_idx_, axis=1)
    #         pm_prune = tf.gather(pm_prune, idx, axis=1, batch_dims=1)
    #         u_survived = tf.gather(hd, prune_idx, axis=1, batch_dims=1)[:, :, tf.newaxis, tf.newaxis]
    #         # print(u_survived.shape)
    #         is_frozen = tf.not_equal(f, 0.5)
    #         x = tf.where(is_frozen, frozen, u_survived)
    #         # print(is_frozen.shape, frozen.shape)
    #
    #         pm_ = tf.where(tf.squeeze(is_frozen, axis=(2, 3)),
    #                        pm + tf.abs(tf.squeeze(llr, axis=(2, 3))) *
    #                        tf.cast(tf.squeeze(tf.not_equal(tf.expand_dims(tf.expand_dims(hd_, -1), -1), frozen),
    #                                           axis=(2, 3)), tf.float32),
    #                        pm_prune)
    #         new_order = tf.tile(tf.expand_dims(tf.range(nL), 0), [ey.shape[0], 1]) \
    #             if f[0, 0, 0, 0] != 0.5 else (prune_idx % nL)
    #         return x, tf.identity(x), llr, pm_, new_order
    #
    #     f_halves = tf.split(f, num_or_size_splits=2, axis=2)
    #     f_left, f_right = f_halves
    #     r_halves = tf.split(r, num_or_size_splits=2, axis=2)
    #     r_left, r_right = r_halves
    #
    #     ey_odd, ey_even = self.split_even_odd_list.call(ey)
    #
    #     # Compute soft mapping back one stage
    #     ey1est = self.checknode.call((ey_odd, ey_even))
    #     shape = ey1est.shape
    #     # ey1est = self.layer_norms[layer_norm_pointer](tf.reshape(ey1est, [-1, shape[-2], shape[-1]]), training=False)
    #     ey1est = tf.reshape(ey1est, shape)
    #     # R_N^T maps u1est to top polar code
    #     uhat1, u1hardprev, llr_uy_left, pm, new_order = self.decode_list(ey1est, f_left, pm,
    #                                                                      N // 2, r_left, sample=sample)
    #
    #     # Using u1est and x1hard, we can estimate u2
    #
    #     ey_odd = tf.gather(ey_odd, new_order, axis=1, batch_dims=1)
    #     ey_even = tf.gather(ey_even, new_order, axis=1, batch_dims=1)
    #
    #     # Using u1est and x1hard, we can estimate u2
    #     u_emb = tf.squeeze(self.UEmbedding(u1hardprev), axis=-2)
    #     ey2est = self.bitnode.call((ey_odd, ey_even, u_emb))
    #     # ey2est = self.layer_norms[layer_norm_pointer](tf.reshape(ey2est, [-1, shape[-2], shape[-1]]), training=False)
    #     # ey2est = tf.reshape(ey2est, shape)
    #
    #     # Using u1est and x1hard, we can estimate u2
    #     uhat2, u2hardprev, llr_uy_right, pm, new_order2 = self.decode_list(ey2est, f_right, pm,
    #                                                                        N // 2, r_right, sample=sample)
    #     uhat1 = tf.gather(uhat1, new_order2, axis=1, batch_dims=1)
    #     llr_uy_left = tf.gather(llr_uy_left, new_order2, axis=1, batch_dims=1)
    #     u1hardprev = tf.gather(u1hardprev, new_order2, axis=1, batch_dims=1)
    #     new_order = tf.gather(new_order, new_order2, axis=1, batch_dims=1)
    #     u = tf.concat([uhat1, uhat2], axis=2)
    #     llr_uy = tf.concat([llr_uy_left, llr_uy_right], axis=2)
    #     x = self.f2_list.call((u1hardprev, u2hardprev))
    #     x = self.interleave_list.call(x)
    #     return u, x, llr_uy, pm, new_order
