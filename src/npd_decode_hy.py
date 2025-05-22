#%% imports
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import numpy as np
import tensorflow as tf
from src.polar import SCEncoder, SCDecoder, PolarCode, SCLDecoder
from src.channels import Ising
from src.generators import info_bits_generator,  custom_dataset
from src.builders import build_neural_polar_decoder_hy_synced
from src.utils import save_args_to_json, load_json, print_config_summary, visualize_synthetic_channels

#%% set configurations
print(f"TF version: {tf.__version__}")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR

eager_mode = False
if eager_mode:
    print("Running in eager mode")
    tf.config.run_functions_eagerly(True)

def get_args():
    parser = argparse.ArgumentParser(description="Train or evaluate Neural Polar Decoder.")
    parser.add_argument("--channel", type=str, choices=["ising"], default="ising",
                        help="Channel type for data generation.")
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
    parser.add_argument("--save_name", type=str, default=None,
                        help="Save name of decoder.")
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

#%% Print the model configuration
print_config_summary(vars(args), title="Args")
print_config_summary(npd_config, title="Neural Polar Decoder")

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
npd = build_neural_polar_decoder_hy_synced(npd_config, input_shape=input_shape, load_path=model_path)
npd.compile()

#%% Create eval dataset by sampling inputs with HY and transmitting through the channel
encoder = SCEncoder(np.arange(args.N).tolist(), info_bits_num=0, decoder=npd.npd_const)
eval_dataset = custom_dataset(args.batch, args.N, encoder, channel)

#%% Evaluate the MI and estimate the synthetic channels
print("code construction:")
npd.evaluate(eval_dataset, steps=args.mc_length, verbose=args.verbose)
mi = np.mean(npd.synthetic_channel_entropy_metric_x.result().numpy() - npd.synthetic_channel_entropy_metric_y.result().numpy())
print(f"Mutual Information: {mi:.6f}")

#%% Visualize the polarization of the synthetic channels
arr = (npd.synthetic_channel_entropy_metric_x.result() -  npd.synthetic_channel_entropy_metric_y.result())
visualize_synthetic_channels(arr, args.save_dir_path)

#%% Set up the sorted reliabilities and info bits
print("Start decoding:")
info_bits_num = np.floor(args.code_rate * args.N).astype(np.int32)
sorted_reliabilities = np.argsort(-arr)
print(f"Info bits num: {info_bits_num}")
print(f"Sorted reliabilities: {sorted_reliabilities}")

#%% Create dataset for decoding
info_bits_dataset = tf.data.Dataset.from_generator(
    lambda: info_bits_generator(args.batch, args.N),
    output_signature=tf.TensorSpec(shape=(args.batch, args.N), dtype=tf.int32)
).prefetch(tf.data.AUTOTUNE)

#%% SC decoder
print("SC decoder:")
encoder = SCEncoder(sorted_reliabilities, info_bits_num=info_bits_num, decoder=npd.npd_const)
decoder = SCDecoder(npd.npd_channel)
polar_code = PolarCode(encoder=encoder,
                       modulator=tf.identity,
                       channel=channel,
                       decoder=decoder)
polar_code.compile()
polar_code.evaluate(info_bits_dataset, steps=args.mc_length, verbose=args.verbose)

res_sc = (polar_code.ber_metric.result().numpy(), polar_code.fer_metric.result().numpy())

#%% SCL decoder
print(f"SCL decoder with list {args.list_num}:")
decoder = SCLDecoder(npd.npd_channel, list_num=args.list_num)
polar_code = PolarCode(encoder=encoder,
                       modulator=tf.identity,
                       channel=channel,
                       decoder=decoder)
polar_code.compile()
polar_code.evaluate(info_bits_dataset, steps=args.mc_length, verbose=args.verbose)
res_scl = (polar_code.ber_metric.result().numpy(), polar_code.fer_metric.result().numpy())

#%% Save the results
res_path = os.path.join(args.save_dir_path, f"results.txt")
with open(res_path, "w") as f:
    f.write("mi: " + str(mi) + "\n")
    f.write("res_sc: " + " ".join(map(str, res_sc)) + "\n")
    f.write("res_scl: " + " ".join(map(str, res_scl)) + "\n")

print(f"Saved MI, SC and SCL results to: {res_path}")