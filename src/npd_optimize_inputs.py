#%% imports
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger
from keras.optimizers import Adam
from src.generators import info_bits_generator
from src.models import NeuralPolarDecoderHondaYamamoto
from src.builders import build_neural_polar_decoder_hy_synced_optimize
from src.callbacks import ReduceLROnPlateauCustom
from src.utils import (save_args_to_json, load_json, print_config_summary, visualize_synthetic_channels,
                       gpu_init, safe_wandb_init)
from src.channels import Ising

#%% set configurations
print(f"TF version: {tf.__version__}")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
gpu_init(allow_growth=True)
eager_mode = False
if eager_mode:
    print("Running in eager mode")
    tf.config.run_functions_eagerly(True)

def get_args():
    parser = argparse.ArgumentParser(description="Train or evaluate Neural Polar Decoder.")
    parser.add_argument("--channel", type=str, choices=["ising", "awgn"], default="ising",
                        help="Channel type for data generation.")
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
    parser.add_argument("--optimizer_estimation_config_path", type=str, default="../configs/optimizer_config.json",
                        help="Path to optimizer configs.")
    parser.add_argument("--optimizer_improvement_config_path", type=str, default="../configs/optimizer_improvement_config.json",
                        help="Path to optimizer configs.")
    parser.add_argument("--save_dir_path", type=str, default="../results/train-iid",
                        help="Path to save trained model weights.")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=2,
                        help="Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch.")
    return parser.parse_args()

args = get_args()
save_args_to_json(args, os.path.join(args.save_dir_path, "args.json"))
npd_config = load_json(args.npd_config_path)
optimizer_estimation_config = load_json(args.optimizer_estimation_config_path)
optimizer_improvement_config = load_json(args.optimizer_improvement_config_path)

os.makedirs(args.save_dir_path, exist_ok=True)
os.makedirs( os.path.join(args.save_dir_path, 'model'), exist_ok=True)

model_full_path = os.path.join(args.save_dir_path, 'model', f"{args.save_name}_full.weights.h5")
model_path = os.path.join(args.save_dir_path, 'model', f"{args.save_name}.weights.h5")
print(f"Model path: {model_path}")

safe_wandb_init(project="npd_publish",
                entity="data-driven-polar-codes",
                tags=["train", "optimized"],
                config=dict(**vars(args),**npd_config))

#%% Print the model configuration

print_config_summary(vars(args), title="Args")
print_config_summary(npd_config, title="Neural Polar Decoder")
print_config_summary(optimizer_estimation_config, title="Optimizer Estimation")
print_config_summary(optimizer_improvement_config, title="Optimizer Improvement")

#%%  Here the channel can be changed to desired one
if args.channel == "ising":
    channel = Ising()
else:
    raise ValueError(f"Invalid channel type: {args.channel}. Choose 'ising'.")

#%% Build the model
input_shape=(
    (args.batch, args.N, 1),  # shape of x
    (args.batch, args.N, 1)   # shape of y
)
npd = build_neural_polar_decoder_hy_synced_optimize(npd_config, input_shape, channel,  load_path=None)
npd.compile(
            opt_est=Adam(learning_rate=optimizer_estimation_config["learning_rate"],
                         beta_1=optimizer_estimation_config["beta_1"],
                         beta_2=optimizer_estimation_config["beta_2"]),
            opt_impr=Adam(learning_rate=optimizer_improvement_config["learning_rate"],
                          beta_1=optimizer_improvement_config["beta_1"],
                          beta_2=optimizer_improvement_config["beta_2"]),)

#%% Create training dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: info_bits_generator(args.batch, args.N),
    output_signature=tf.TensorSpec(shape=(args.batch, args.N), dtype=tf.float32)
).prefetch(tf.data.AUTOTUNE)

#%% Train the model
lr_scheduler_est = ReduceLROnPlateauCustom(monitor='ce_y',
                                           factor=optimizer_estimation_config['factor'],
                                           patience=optimizer_estimation_config['patience'],
                                           verbose=args.verbose,
                                           min_lr=optimizer_estimation_config['min_lr'],
                                           mode=optimizer_estimation_config['mode'],
                                           optimizer=npd.opt_est)
lr_scheduler_improve = ReduceLROnPlateauCustom(monitor='mi',
                                               factor=optimizer_improvement_config['factor'],
                                               patience=optimizer_improvement_config['patience'],
                                               verbose=args.verbose,
                                               min_lr=optimizer_improvement_config['min_lr'],
                                               mode=optimizer_improvement_config['mode'],
                                               optimizer=npd.opt_improve)
history = npd.fit(train_dataset,
                  epochs=args.epochs,
                  steps_per_epoch=args.steps_per_epoch,
                  callbacks=[lr_scheduler_est, lr_scheduler_improve, WandbMetricsLogger()], verbose=args.verbose)

#%% Save all weights, including the auxiliary RNN input
os.makedirs(os.path.dirname(model_full_path), exist_ok=True)
npd.save_weights(f"{model_full_path}")

#%% Visualize the polarization of the synthetic channels
arr = (npd.synthetic_channel_entropy_metric_x.result() -  npd.synthetic_channel_entropy_metric_y.result())
visualize_synthetic_channels(arr, args.save_dir_path)

#%% Discard the RNN inputs model
npd_hy = NeuralPolarDecoderHondaYamamoto(npd.npd_const, npd.npd_channel)
npd_hy.build(input_shape)
npd_hy.compile()

#%% Save the model without the RNN inputs
npd_hy((tf.zeros(shape=(args.batch, args.N, 1), dtype=tf.int32),
             tf.zeros(shape=(args.batch, args.N, 1), dtype=tf.float32)))
npd_hy.save_weights(f"{model_path}")

print("Training complete. Model saved to:", model_path)

