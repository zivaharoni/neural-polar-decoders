#%%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import numpy as np
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger
from src.channels import AWGN, Ising, BPSK
from src.polar import PolarEncoder, SCDecoder, PolarCode, SCLDecoder
from src.generators import info_bits_generator, iid_awgn_generator, iid_ising_generator
from src.builders import build_neural_polar_decoder_iid_synced
from src.utils import (save_args_to_json, load_json, print_config_summary, visualize_synthetic_channels,
                       gpu_init, safe_wandb_init)

#%% set configurations
print(f"TF version: {tf.__version__}")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu_init(allow_growth=True)

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
    parser.add_argument("--list_num", type=int, default=8,
                        help="List size in SCL.")
    parser.add_argument("--code_rate", type=float, default=0.25,
                        help="Code rate.")
    parser.add_argument("--mc_length", type=int, default=1000,
                        help="MC length used for evaluation.")
    parser.add_argument("--save_name", type=str, default="model",
                        help="Model name used for saving.")
    parser.add_argument("--npd_config_path", type=str, default="../configs/npd_small_config.json",
                        help="Path to npd configs.")
    parser.add_argument("--save_dir_path", type=str, default="../results/train-iid",
                        help="Path to save trained model weights.")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=2,
                        help="Verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch.")
    return parser.parse_args()

args = get_args()
save_args_to_json(args, os.path.join(args.save_dir_path, "args.json"))
npd_config = load_json(args.npd_config_path)

os.makedirs(args.save_dir_path, exist_ok=True)
model_path = os.path.join(args.save_dir_path, 'model', f"{args.save_name}.weights.h5")
print(f"Model path: {model_path}")

safe_wandb_init(project="npd_publish",
                entity="data-driven-polar-codes",
                tags=["decode", "iid"],
                config=dict(**vars(args),**npd_config))

#%% Print the model configuration
print_config_summary(vars(args), title="Args")
print_config_summary(npd_config, title="Neural Polar Decoder")

#%% Here the channel can be changed to desired one
if args.channel == "ising":
    channel = Ising()
    modulator = tf.identity
    data_gen = iid_ising_generator(args.batch, args.N, p=0.5)
elif args.channel == "awgn":
    channel = AWGN(snr_db=args.snr_db)
    modulator = BPSK()
    data_gen = iid_awgn_generator(args.batch, args.N, snr_db=args.snr_db, p=0.5)
else:
    raise ValueError(f"Invalid channel type: {args.channel}. Choose 'ising' or 'awgn'.")

#%%
input_shape=(
    (args.batch, args.N, 1),  # shape of x
    (args.batch, args.N, 1)   # shape of y
)
npd = build_neural_polar_decoder_iid_synced(npd_config, input_shape, load_path=model_path)
npd.compile()

#%% Create dataset for evaluating the reliability of the synthetic channels.
construction_dataset = tf.data.Dataset.from_generator(
    lambda: iid_ising_generator(args.batch, args.N),
    output_signature=(
        tf.TensorSpec(shape=(args.batch, args.N, 1), dtype=tf.int32),
        tf.TensorSpec(shape=(args.batch, args.N, 1), dtype=tf.float32)
    )
).prefetch(tf.data.AUTOTUNE)

#%% Evaluate the MI and estimate the synthetic channels
print("code construction:")
npd.evaluate(construction_dataset, steps=args.mc_length, verbose=args.verbose, callbacks=[WandbMetricsLogger()])
mi = 1.0 - np.mean(npd.synthetic_channel_entropy_metric.result().numpy())
wandb.summary["mi"] = mi
#%% Visualize the polarization of the synthetic channels
arr = 1.0 - npd.synthetic_channel_entropy_metric.result()
visualize_synthetic_channels(arr, args.save_dir_path)

#%% Set up the sorted reliabilities and info bits
info_bits_num = np.floor(args.code_rate * args.N).astype(np.int32)
sorted_reliabilities = np.argsort(npd.synthetic_channel_entropy_metric.result())
print(f"Info bits num: {info_bits_num}")
print(f"Sorted reliabilities: {sorted_reliabilities}")

#%% Create dataset for decoding
print("Start decoding:")
info_bits_dataset = tf.data.Dataset.from_generator(
    lambda: info_bits_generator(args.batch, info_bits_num),
    output_signature=tf.TensorSpec(shape=(args.batch, info_bits_num), dtype=tf.int32)
).prefetch(tf.data.AUTOTUNE)

#%% SC decoder
print("SC decoder:")
encoder = PolarEncoder(sorted_reliabilities, info_bits_num)
decoder = SCDecoder(npd)
polar_code = PolarCode(encoder=encoder,
                       modulator=modulator,
                       channel=channel,
                       decoder=decoder)
polar_code.compile()

polar_code.evaluate(info_bits_dataset, steps=args.mc_length, verbose=args.verbose, callbacks=[WandbMetricsLogger()])
res_sc = (polar_code.ber_metric.result().numpy(), polar_code.fer_metric.result().numpy())
wandb.summary["ber_sc"] = res_sc[0]
wandb.summary["fer_sc"] = res_sc[1]

#%% SCL decoder
print(f"SCL decoder with list {args.list_num}:")
decoder = SCLDecoder(npd, list_num=args.list_num)
polar_code = PolarCode(encoder=encoder,
                       modulator=modulator,
                       channel=channel,
                       decoder=decoder)

polar_code.compile()
polar_code.evaluate(info_bits_dataset, steps=args.mc_length, verbose=args.verbose, callbacks=[WandbMetricsLogger()])
res_scl = (polar_code.ber_metric.result().numpy(), polar_code.fer_metric.result().numpy())
wandb.summary["ber_scl"] = res_scl[0]
wandb.summary["fer_scl"] = res_scl[1]

#%% Save the results
res_path = os.path.join(args.save_dir_path, f"results.txt")

with open(res_path, "w") as f:
    f.write("mi: " + str(mi) + "\n")
    f.write("res_sc: " + " ".join(map(str, res_sc)) + "\n")
    f.write("res_scl: " + " ".join(map(str, res_scl)) + "\n")

print(f"Saved MI, SC and SCL results to: {res_path}")