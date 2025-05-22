#%% imports
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from src.generators import iid_awgn_generator, iid_ising_generator
from src.builders import build_neural_polar_decoder_iid_synced
from src.callbacks import ReduceLROnPlateauCustom
from src.utils import save_args_to_json, load_json, print_config_summary, visualize_synthetic_channels

#%% set configurations
print(f"TF version: {tf.__version__}")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

eager_mode = False
if eager_mode:
    print("Running in eager mode")
    tf.config.run_functions_eagerly(True)

def get_args():
    parser = argparse.ArgumentParser(description="Train or evaluate Neural Polar Decoder.")
    parser.add_argument("--channel", type=str, choices=["ising", "awgn"], default="ising",
                        help="Channel type for data generation.")
    parser.add_argument("--snr_db", type=float, default=0.0,
                        help="SNR in dB for AWGN channel.")
    parser.add_argument("--batch", type=int, default=100,
                        help="Batch size.")
    parser.add_argument("--N", type=int, default=64,
                        help="Block length.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--steps_per_epoch", type=int, default=1000,
                        help="Number of steps per training epoch.")
    parser.add_argument("--mc_length", type=int, default=1000,
                        help="MC length used for evaluation.")
    parser.add_argument("--save_name", type=str, default="model",
                        help="Model name used for saving.")
    parser.add_argument("--npd_config_path", type=str, default="../configs/npd_small_config.json",
                        help="Path to npd configs.")
    parser.add_argument("--optimizer_config_path", type=str, default="../configs/optimizer_config.json",
                        help="Path to optimizer configs.")
    parser.add_argument("--save_dir_path", type=str, default="../results/train-iid",
                        help="Path to save trained model weights.")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=2,
                        help="Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch.")
    return parser.parse_args()

args = get_args()
save_args_to_json(args, os.path.join(args.save_dir_path, "args.json"))
npd_config = load_json(args.npd_config_path)
optimizer_config = load_json(args.optimizer_config_path)

os.makedirs(args.save_dir_path, exist_ok=True)
os.makedirs( os.path.join(args.save_dir_path, 'model'), exist_ok=True)
model_path = os.path.join(args.save_dir_path, 'model', f"{args.save_name}.weights.h5")

#%% Print the model configuration

print_config_summary(vars(args), title="Args")
print_config_summary(npd_config, title="Neural Polar Decoder")

#%%  Here the channel can be changed to desired one
if args.channel == "ising":
    data_gen = iid_ising_generator(args.batch, args.N, p=0.5)
elif args.channel == "awgn":
    data_gen = iid_awgn_generator(args.batch, args.N, snr_db=args.snr_db, p=0.5)
else:
    raise ValueError(f"Invalid channel type: {args.channel}. Choose 'ising' or 'awgn'.")

#%% Build the model
input_shape=(
    (args.batch, args.N, 1),  # shape of x
    (args.batch, args.N, 1)   # shape of y
)

npd = build_neural_polar_decoder_iid_synced(npd_config, input_shape, load_path=None)
npd.compile(optimizer=Adam(learning_rate=optimizer_config["learning_rate"],
                           beta_1=optimizer_config["beta_1"],
                           beta_2=optimizer_config["beta_2"]))

#%% Create training dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_gen,
    output_signature=(
        tf.TensorSpec(shape=(args.batch, args.N, 1), dtype=tf.int32),
        tf.TensorSpec(shape=(args.batch, args.N, 1), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

#%% Train the model
lr_scheduler = ReduceLROnPlateauCustom(monitor='ce', factor=optimizer_config["factor"],
                                       patience=optimizer_config["patience"], verbose=args.verbose,
                                       min_lr=optimizer_config["min_lr"], mode=optimizer_config["mode"])
history = npd.fit(train_dataset,
                  epochs=args.epochs,
                  steps_per_epoch=args.steps_per_epoch,
                  callbacks=[lr_scheduler], verbose=args.verbose)
print("Training complete.")

#%% MC evaluation of the ce rate
npd.evaluate(train_dataset, steps=args.mc_length, verbose=args.verbose)

#%% Visualize the polarization of the synthetic channels
ce = npd.synthetic_channel_entropy_metric.result()
arr = np.ones_like(ce) - ce
visualize_synthetic_channels(arr, args.save_dir_path)

#%% Save model weights
npd.save_weights(model_path)  # creates

print("Training complete. Model saved to:", model_path)
