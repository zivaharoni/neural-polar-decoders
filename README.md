# Neural Polar Decoders

This repository contains the code for training and evaluating neural polar decoders (NPDs) on communication channels. 
It includes a code that optimizes the code rate of the polar code by maximizing the mutual information (MI) of the channel's inputs and outputs.
---

## Project Structure

<pre>
neural-polar-decoders/
├── configs/                                # JSON configs for model and optimizer
│   ├── npd_small_config.json               # Small NPD architecture
│   ├── npd_medium_config.json              # Medium NPD architecture
│   ├── npd_optimizer_config.json           # Config for NPD estimation optimizer
│   └── optimizer_improvement_config.json   # Config for NPD improvement optimizer
│
├── results/                                # Output directory for runs
│   └── TIMESTAMP/                          # Auto-created directory for each run
│       ├── args.json                       # Command line arguments
│       ├── results.txt                     # MI and decoding results
│       ├── synthetic_channels.png        # Polarization of synthetic channels
│       └── model/                          # Saved weights
│
├── src/                                    # Main source files
│   ├── builders.py                         # Model constructor functions
│   ├── callbacks.py                        # Custom callbacks for training
│   ├── channels.py                         # Channel classes
│   ├── generators.py                       # Data generators
│   ├── input_distribution.py               # Input distribution class for code rate optimization
│   ├── layers.py                           # Custom layers for NPD
│   ├── models.py                           # NPD model training classes
│   ├── polar.py                            # Polar code classes
│   ├── sc_ops.py                           # SC decoding operations
│   ├── utils.py                            # Utility functions
│   ├── npd_train_iid.py                    # Training for uniform and iid input distribution
│   ├── npd_decode_iid.py                   # Evaluation for uniform and iid input distribution
│   ├── npd_optimize_inputs.py              # Training for input distribution optimization
│   └── npd_decode_hy.py                    # Evaluation for optimized input distribution
│
├── scripts/                               # Bash scripts for training and evaluation
│   ├── run_npd_eval.sh             # Bash launcher for evaluation
│   ├── run_npd_eval_hy.sh          # Bash launcher for hybrid evaluation
│   ├── run_npd_eval_sc.sh          # Bash launcher for SC evaluation
│   └── run_npd_train.sh            # Bash launcher for training
│
├── requirements.txt             # (optional) Python dependencies
├── LICENSE                      # License file
└── README.md                    # You are here 

</pre>

---

## Setup

For conda:

```bash
  conda create -n npd-env python=3.9 -y
  conda activate npd-env
```
Clone the repository:
```bash
    git clone neural-polar-decoders.git
    cd  neural-polar-decoders 
```

Install dependencies:

```bash
pip install -r requirements.txt
```



---

## Training an NPD for uniform and iid input distribution

Train an NPD and evaluate on the Ising channel with code rate 0.4 and list size 8:
```bash
bash ./runfiles/train-and-evaluate-iid-inputs.sh \
    --channel ising \
    --N 32 \
    --batch 64 \
    --model_size small \
    --epochs 5	\
    --steps_per_epoch 100 \
    --mc_length 100 \
    --code_rate 0.4 \
    --list_num 8
```

## Training an NPD with optimized input distribution

Train and evaluate an NPD on the Ising channel:
```bash

bash ./runfiles/train-and-evaluate-npd-optimize-inputs.sh \
	--channel ising \
	--N 32 \
	--batch 64 \
	--model_size medium \
	--epochs 5	
```

Model and logs are saved under:

```
results/<save_dir>/<timestamp>/
```

## Notes

For testing on other channels, you can change the `--channel` argument and implement a new channel class in `src/channels.py`

---

## Citation

This repo is part of a series of research papers on learning-based polar decoding.
(Citation to be added)

---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

---
## Contact

Ziv Aharoni
Postdoctoral Associate, Duke University
ziv.aharoni at duke.edu