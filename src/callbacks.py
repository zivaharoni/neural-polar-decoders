import tensorflow as tf
import keras

class ReduceLROnPlateauCustom(tf.keras.callbacks.Callback):
    def __init__(self, monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1, mode='min', optimizer=None):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.mode = mode
        self.best = None
        self.wait = 0
        self.monitor_op = tf.math.less if mode == 'min' else tf.math.greater
        self.metric = keras.metrics.Mean(name=monitor)
        self.specified_optimizer = optimizer

    def on_epoch_begin(self, epoch, logs=None):
        self.metric.reset_state()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        self.metric.update_state(current)

    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.specified_optimizer if self.specified_optimizer is not None else self.model.optimizer

        current = self.metric.result()
        if current is None:
            return

        if self.best is None or self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = tf.identity(optimizer.learning_rate)
                if old_lr > self.min_lr:
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    optimizer.learning_rate.assign(new_lr)
                    if self.verbose:
                        print(f"Epoch {epoch+1}: {self.monitor} did not improve over {self.best:.4f}, reducing LR to {new_lr:.2e}")
                self.wait = 0
