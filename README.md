# Neural Polar Decoders

This repository contains the code for training and evaluating neural polar decoders (NPDs) on communication channels. 
It includes a code that optimizes the code rate of the polar code by maximizing the mutual information (MI) of the channel's inputs and outputs.

The code is based on the paper "Data-Driven Neural Polar Decoders for Unknown Channels with and without Memory" [[1]](https://ieeexplore.ieee.org/document/10711969) and the code rate optimization presented in "Code Rate Optimization via Neural Polar Decoders" [[2]](https://ieeexplore.ieee.org/document/10619429).

---

## Project Structure

<pre>
neural-polar-decoders/
├── configs/                                    # JSON configs for model and optimizer
│   ├── npd_small_config.json                   # Small NPD architecture
│   ├── npd_medium_config.json                  # Medium NPD architecture
│   ├── npd_optimizer_config.json               # Config for NPD estimation optimizer
│   └── optimizer_improvement_config.json       # Config for NPD improvement optimizer
│
├── results/                                    # Output directory for runs
│   └── TIMESTAMP/                              # Auto-created directory for each run
│       ├── args.json                           # Command line arguments
│       ├── results.txt                         # MI and decoding results
│       ├── synthetic_channels.png              # Polarization of synthetic channels
│       └── model/                              # Saved weights
│
├── src/                                        # Main source files
│   ├── builders.py                             # Model constructor functions
│   ├── callbacks.py                            # Custom callbacks for training
│   ├── channels.py                             # Channel classes
│   ├── generators.py                           # Data generators
│   ├── input_distribution.py                   # Input distribution class for code rate optimization
│   ├── layers.py                               # Custom layers for NPD
│   ├── models.py                               # NPD model training classes
│   ├── polar.py                                # Polar code classes
│   ├── sc_ops.py                               # SC decoding operations
│   ├── utils.py                                # Utility functions
│   ├── npd_train_iid.py                        # Training for uniform and iid input distribution
│   ├── npd_decode_iid.py                       # Evaluation for uniform and iid input distribution
│   ├── npd_optimize_inputs.py                  # Training for input distribution optimization
│   └── npd_decode_hy.py                        # Evaluation for optimized input distribution
│
├── scripts/                                    # Bash scripts for training and evaluation
│   ├── train-and-evaluate-iid-inputs.sh        # Bash launcher for iid input distribution
│   └── train-and-evaluate-optimize-inputs.sh   # Bash launcher for input distribution optimization
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
    git clone https://github.com/zivaharoni/neural-polar-decoders.git
    cd  neural-polar-decoders 
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training an NPD for uniform and i.i.d. input distribution

Train an NPD and evaluate on the Ising channel with code rate 0.4 and list size 8:
```bash
bash ./runfiles/train-and-evaluate-iid-inputs.sh --channel ising --N 1024 --batch 256 --model_size small --epochs 100	\
  --steps_per_epoch 1000 --mc_length 10000 --code_rate 0.4 --list_num 32
```


## Training an NPD with optimized input distribution

Train and evaluate an NPD on the Ising channel:
```bash
bash ./runfiles/train-and-evaluate-optimize-inputs.sh --channel ising --N 1024 --batch 256 --model_size small \
  --epochs 1000	--steps_per_epoch 1000 --mc_length 10000 --code_rate 0.4 --list_num 32 --threshold 0.0
```

Model and logs are saved under:

```
results/<save_dir>/<timestamp>/
```


## Results and Hyperparameters

**Hyperparameters**

|  N   | MC length | Batch | Model Size | Epochs | Steps/Epoch | Code Rate | List Size | Threshold |
|:----:|:---------:|:-----:|:----------:|:------:|:-----------:|:---------:|:---------:|:---------:|
| 1024 |   10000   |  256  |   small    |  1000  |    1000     |    0.4    |    256    |   0.25    |

**Results**

| Input Distribution |  N   |   MI    | SC FER | SCL FER |
|:------------------:|:----:|:-------:|:------:|:-------:|
|        iid         | 1024 | 0.4502  | 0.648  | 0.1900  |
|     optimized      | 1024 | 0.541  | 0.908  | 0.023   |


## Notes

For testing on other channels, you can change the `--channel` argument and implement a new channel class in `src/channels.py`

---

## Citation

This repo is part of a series of research papers on learning-based polar decoding.
Data-driven neural polar decoders are introduced in [[1]](https://ieeexplore.ieee.org/document/10711969), and the code rate optimization is presented in [[2]](https://ieeexplore.ieee.org/document/10619429).
```latex
@article{aharoniDatadrivenNeuralPolar2024,
  title = {Data-Driven {{Neural Polar Decoders}} for {{Unknown Channels}} with and without {{Memory}}},
  author = {Aharoni, Ziv and Huleihel, Bashar and Pfister, Henry D and Permuter, Haim H},
  year = {2024},
  journal = {IEEE Transactions on Information Theory},
  publisher = {IEEE},
  keywords = {Artificial neural networks,Channel estimation,Channel models,Channels with memory,Computational complexity,data-driven,Decoding,Memoryless systems,neural polar decoder,polar codes,Polar codes,Power capacitors,Training,Transforms}
}

@inproceedings{aharoniCodeRateOptimization2024,
  title = {Code {{Rate Optimization}} via {{Neural Polar Decoders}}},
  booktitle = {2024 {{IEEE International Symposium}} on {{Information Theory}} ({{ISIT}})},
  author = {Aharoni, Ziv and Huleihel, Bashar and Pfister, Henry D. and Permuter, Haim H.},
  year = {2024},
  pages = {2424--2429},
  issn = {2157-8117},
  doi = {10.1109/ISIT57864.2024.10619429},
  keywords = {Channel capacity,Channel models,channels with memory,Codes,Complexity theory,data-driven,Decoding,Knowledge engineering,Memoryless systems,polar codes,Power capacitors}
}

```
---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

---
## Contact

Ziv Aharoni
Postdoctoral Associate, Duke University
ziv.aharoni at duke.edu
