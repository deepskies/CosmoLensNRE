import subprocess
import numpy as np
import os

num_lenses = 500000

 # Generate random seed
seed = np.random.randint(0, 100)

Om_min = 0.1
Om_max = 0.9
w_min = -1.8
w_max = -0.35

dirname = "entire_range_narrowprior_copy1"

if not os.path.exists(f"/deepskieslab/stronglensing/hsbi/datasets/train_data/{dirname}"):
    os.mkdir(f"/deepskieslab/stronglensing/hsbi/datasets/train_data/{dirname}")

# Set name and outdir
name = f"/deepskieslab/stronglensing/hsbi/datasets/train_data/{dirname}/"
outdir = f"/deepskieslab/stronglensing/hsbi/datasets/train_data/{dirname}/"

# Construct the command to run generate_data.py
command = [
    "python", "generate_train_data_narrowprior.py",
    "--name", name,
    "--outdir", outdir,
    "--seed", str(seed),
    "--num", str(num_lenses),
    "--Om_min", str(Om_min),
    "--Om_max",str(Om_max),
    "--w_min", str(w_min),
    "--w_max",str(w_max)
]

# Run the command
subprocess.run(command)
