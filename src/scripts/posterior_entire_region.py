"""
Script to estimate the posterior distribution of the cosmological parameters from the test data.
"""
import subprocess
import os

resolution = 25

ommin = 0.1
ommax = 0.9
wmin = -1.8
wmax = -0.4
num_cosmologies = 100
alpha = 0.175255
beta = 0.11896

modelname = "entire_region_narrowprior"
datadir = "entire_region_narrowprior_wom"

if not os.path.exists(f"/deepskieslab/stronglensing/hsbi/models/plots/evaluate/{modelname}"):
    os.mkdir(f"/deepskieslab/stronglensing/hsbi/models/plots/evaluate/{modelname}")

if not os.path.exists(f"/deepskieslab/stronglensing/hsbi/models/plots/evaluate/{modelname}/{datadir}"):
    os.mkdir(f"/deepskieslab/stronglensing/hsbi/models/plots/evaluate/{modelname}/{datadir}")


command = [
        'python', 'evaluate_with_calibration.py',
        '--model_path', '/deepskieslab/stronglensing/hsbi/models/',
        '--model_name', modelname,
        '--data_dir', datadir,
        '--output_path', f'/deepskieslab/stronglensing/hsbi/models/plots/evaluate/{modelname}/{datadir}/',
        '--w_min', str(wmin),
        '--w_max', str(wmax),
        '--om_min', str(ommin),
        '--om_max',str(ommax),
        '--n_lenses', str(num_cosmologies), 
    '--alpha', str(alpha),
    '--beta', str(beta)
    ]

# Run the command
subprocess.run(command)
