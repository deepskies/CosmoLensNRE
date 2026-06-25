"""
Script to generate test data for the NRE model. Generates a set of lens datasets with cosmological parameters (w, Om) sampled from specified ranges. 
"""
import subprocess
import numpy as np
import os

num_lenses = 100
num_cosmologies = 100

ommin = 0.22
ommax = 0.77
wmin = -1.70
wmax = -0.50

dirname = "entire_region_narrowprior_wom"

w_vals = np.random.uniform(wmin, wmax, num_cosmologies)
Om_vals = np.random.uniform(ommin, ommax, num_cosmologies)

if not os.path.exists(f"/deepskieslab/stronglensing/hsbi/datasets/test_data/{dirname}"):
    os.mkdir(f"/deepskieslab/stronglensing/hsbi/datasets/test_data/{dirname}")


for i, (w0,Om0) in enumerate(zip(w_vals,Om_vals)):
    print(f"w: {w0:.2f}, Om: {Om0:.2f}")
    # Generate random seed
    seed = np.random.randint(0, 100)
    
    # Set name and outdir
    name = f"/deepskieslab/stronglensing/hsbi/datasets/test_data/{dirname}/data_{i}"
    outdir = f"/deepskieslab/stronglensing/hsbi/datasets/test_data/{dirname}/data_{i}"
    
    # Construct the command to run generate_data.py
    command = [
        "python", "generate_data_one_cosmology.py",
        "--name", name,
        "--outdir", outdir,
        "--seed", str(seed),
        "--num", str(num_lenses),
        "--Om0", str(Om0),
        "--w0", str(w0)
    ]
    
    # Run the command
    subprocess.run(command)
