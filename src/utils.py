import os
import json
import wandb
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.metrics import Metric

class MeanTensor(Metric):
    def __init__(self, name="mean_tensor", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = None
        self.count = None

    def update_state(self, values, sample_weight=None):
        values = tf.convert_to_tensor(values, dtype=tf.float32)

        if self.total is None:
            self.total = self.add_weight(
                name="total",
                shape=values.shape[1:],  # exclude batch dimension
                initializer="zeros"
            )
            self.count = self.add_weight(
                name="count",
                shape=(),
                initializer="zeros"
            )

        values_sum = tf.reduce_sum(values, axis=0)  # sum over batch
        batch_size = tf.cast(tf.shape(values)[0], tf.float32)

        self.total.assign_add(values_sum)
        self.count.assign_add(batch_size)

    def result(self):
        return self.total / self.count

    def reset_states(self):
        if self.total is not None:
            self.total.assign(tf.zeros_like(self.total))
            self.count.assign(0.0)


def save_args_to_json(args, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved args to: {save_path}")

def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def print_config_summary(config, title="Configuration Summary"):
    def print_dict(d, indent=2):
        for k, v in d.items():
            print(" " * indent + f"{k:<20}: {v}")

    print(f"{title}".center(50, "="))
    print_dict(config)
    print("=" * 50)

def visualize_synthetic_channels(arr, save_path):
    arr = arr.numpy()
    fig = plt.figure()
    plt.scatter(np.arange(arr.shape[0]), arr, s=2)  # s is marker size
    plt.xlabel("Channel Index")
    plt.ylabel("$I(U_i; Y^N|U^{i-1})$")
    plt.title("Synthetic Channels")
    plt.grid(True)
    # Save plot
    plot_path = os.path.join(save_path, f"synthetic_channels.png")
    plt.savefig(plot_path, bbox_inches='tight')
    wandb.log({"synthetic_channels": wandb.Image(fig)})
    plt.close()
    print(f"Saved polarization to: {plot_path}")

def gpu_init(allow_growth=True):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, allow_growth)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found. Using CPU.")