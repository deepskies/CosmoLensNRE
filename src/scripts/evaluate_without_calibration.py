"""
Evaluate the trained NRE on a set of test lens datasets using uncalibrated MCMC posteriors.

Identical in structure to evaluate_with_calibration.py, but the log-likelihood is the raw
sum of NRE log-ratios with no correction.

Outputs per run:
  - Parity plot: posterior mean vs. true (w, Om) with fractional residuals and reduced chi-square
  - Coverage plot: fraction of lenses with true value inside X% CI vs. X% (calibration diagnostic)
  - Corner plot grid: joint (Om, w) posteriors for up to the first 25 lenses
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import emcee
import tensorflow as tf
import time
import gc
from cycler import cycler
from concurrent.futures import ThreadPoolExecutor
from mpl_toolkits.axes_grid1 import make_axes_locatable
import corner

# define your plot style
best_style = {
    "font.family": "sans-serif",
    "mathtext.fontset": "custom",
    "mathtext.rm": "TeX Gyre Heros",
    "mathtext.bf": "TeX Gyre Heros:bold",
    "mathtext.sf": "TeX Gyre Heros",
    "mathtext.it": "TeX Gyre Heros:italic",
    "mathtext.tt": "TeX Gyre Heros",
    "mathtext.cal": "TeX Gyre Heros",
    "mathtext.default": "regular",
    "figure.figsize": (10.0, 10.0),
    "font.size": 26,
    "axes.labelsize": "medium",
    "axes.unicode_minus": False,
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "legend.fontsize": "small",
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.5,
    "xtick.direction": "in",
    "xtick.major.size": 12,
    "xtick.minor.size": 6,
    "xtick.major.pad": 6,
    "xtick.top": True,
    "xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.top": True,
    "xtick.minor.bottom": True,
    "xtick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.major.size": 12,
    "ytick.minor.size": 6.0,
    "ytick.right": True,
    "ytick.major.left": True,
    "ytick.major.right": True,
    "ytick.minor.left": True,
    "ytick.minor.right": True,
    "ytick.minor.visible": True,
    "grid.alpha": 0.8,
    "grid.linestyle": ":",
    "axes.linewidth": 2,
    "savefig.transparent": False,
}
plt.style.use(best_style)
cols = ["#DB4437", "#4285F4", "#0F9D58", "#F4B400", "purple", "goldenrod", "peru",
        "coral", "turquoise", 'gray', 'navy', 'm', 'darkgreen', 'fuchsia', 'steelblue']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cols)

class PosteriorCoverage:
    """Runs uncalibrated NRE-MCMC for a single test dataset and returns posterior samples."""
    def __init__(self, model_path, model_name, data_path, w_column_name, om_column_name, zl_column_name, zs_column_name, v_column_name, w_min, w_max, om_min, om_max):
        self.model_path = model_path
        self.model_name = model_name
        self.data_path = data_path
        self.w_column_name = w_column_name
        self.om_column_name = om_column_name
        self.zl_column_name = zl_column_name
        self.zs_column_name = zs_column_name
        self.v_column_name = v_column_name
        self.w_min = w_min
        self.w_max = w_max
        self.om_min = om_min
        self.om_max = om_max
        self.model = tf.keras.models.load_model(model_path + model_name + '.keras')

    def get_logr_mcmc(self, images, aparam, sample_theta):
        ''''
        Function to predict the log likelihood-to-evidence ratio (logr) of all the test data at a time

        Input:
        model: The trained model 
        images: test images
        sample_theta: a list of theta values to compute logr for

        Output:
        log r 
        '''
        theta_array = np.array([sample_theta]*images.shape[0])
        output = self.model.predict([images, aparam, theta_array], verbose=0).flatten()
        return output

    def log_prior(self, theta, w_low=-1.5, w_high=-0.5, om_low=0.1, om_high=0.5):
        """
        prior for w, om
        """
        w, om = theta
        if w_low < w < w_high and om_low < om < om_high:
            return 0.0
        return -np.inf

    def log_likelihood(self, theta, data, aparam, w_low, w_high, om_low, om_high):
        """
        Calculate the log likelihood + log prior.
        Uses the raw sum of NRE log-ratios as the likelihood with no affine correction.
        """
        lp = self.log_prior(theta, w_low, w_high, om_low, om_high)
        if not np.isfinite(lp):
            return -np.inf
        logr_array = self.get_logr_mcmc(data, aparam, theta)
        ll = np.sum(logr_array)
        return ll + lp

    def get_posterior_mcmc(self, data, aparam, walkers=10, nsteps=10000, nburn=200, initial_w=-1.0, initial_om=0.3, multithread=False):
        """
        MCMC sampling

        Output:
        Sampler and Samples
        """
        pos = np.array([initial_w, initial_om]) + np.array([initial_w, initial_om]) * 1e-3 * np.random.randn(walkers, 2)
        nwalkers, ndim = pos.shape

        if multithread:
            POOL = ThreadPoolExecutor(20)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_likelihood, args=(data, aparam, self.w_min, self.w_max, self.om_min, self.om_max), pool=POOL)

            print("Running first burn-in...")
            pos, lp, _ = sampler.run_mcmc(pos, nburn, progress=True)
            print('Max lp @', pos[np.argmax(lp)])

            print("Running production...")
            pos = pos[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
            sampler.reset()
            pos, lp, _ = sampler.run_mcmc(pos, nsteps, progress=True)
            print('Max lp @', pos[np.argmax(lp)])
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_likelihood, args=(data, aparam, self.w_min, self.w_max, self.om_min, self.om_max))
            print("Running first burn-in...")
            pos, lp, _ = sampler.run_mcmc(pos, nburn, progress=True)
            print('Max lp @', pos[np.argmax(lp)])

            print("Running production...")
            pos = pos[np.argmax(lp)] + 1e-4 * np.random.randn(nwalkers, ndim)
            sampler.reset()
            pos, lp, _ = sampler.run_mcmc(pos, nsteps, progress=True)
            print('Max lp @', pos[np.argmax(lp)])

        flat_samples = sampler.get_chain(discard=int(nsteps / 4), flat=True)

        return sampler, flat_samples

    def get_samples_wom(self, image_dir, walkers=5, nsteps=1000, nburn=200):
    
        # read the data
        print('Load the data...')
        images = np.load(image_dir + '/CONFIGURATION_1_images.npy', allow_pickle=True)
        metadata = pd.read_csv(image_dir + '/CONFIGURATION_1_metadata.csv')
        
        # preprocess the data
        images = np.einsum('lkij->lijk', images)
        # theta = metadata[[self.w_column_name, self.om_column_name]].to_numpy().astype(np.float32)
        aparms = metadata[[self.zl_column_name, self.zs_column_name, self.v_column_name]].to_numpy().astype(np.float32)

        edge_size = images.shape[1]
        images = edge_size * edge_size * (images / np.sum(images, axis=(1, 2), keepdims=True))
        images = images.reshape(images.shape[0], -1)
        images = images.reshape(images.shape[0], edge_size, edge_size, 1)

        # return true values
        true_w = metadata[self.w_column_name].unique()[0]
        true_om = metadata[self.om_column_name].unique()[0]

        print(true_w, true_om)

        sampler, flat_samples = self.get_posterior_mcmc(images, aparms, walkers=walkers, nsteps=nsteps, nburn=nburn, 
                                               initial_w=-1.5, initial_om=0.31, multithread=False)
        samples_w = flat_samples[:, 0]
        samples_om = flat_samples[:, 1]

        return flat_samples, samples_w, samples_om, true_w, true_om

def calculate_coverage_fraction_density_estimator(n_lenses, sampled_values_list, truth_array, percentile_list):
    # sample from posterior probability
    count_array = []
    for i in range(0, n_lenses):
        sampled_values = sampled_values_list[i]
        count_vector = []
        # print('pair: ', i+1)
        # print('truth : ', truth_array[i])
        for ind, cov in enumerate(percentile_list):
            percentile_l = 50.0 - cov/2
            percentile_u = 50.0 + cov/2 
            confidence_l = np.percentile(sampled_values,percentile_l,axis=0)
            confidence_u = np.percentile(sampled_values,percentile_u,axis=0)
            # print(confidence_l, confidence_u)
            count = np.logical_and(confidence_u - truth_array[i] > 0, truth_array[i] - confidence_l > 0)
            count_vector.append(count)
            # print(count)
        # print(i, count_vector)
        count_array.append(count_vector)
    count_sum_array = np.sum(count_array, axis=0)
    # print(count_sum_array)
    frac_lens_within_vol = np.array(count_sum_array)
    return frac_lens_within_vol/n_lenses

def plot_joint_with_marginals(
    flat_samples_list,
    samples_w_list,
    samples_om_list,
    true_w_list,
    true_om_list,
    n=None,
    color="steelblue",    
    figsize_per_cell=3.2,
    bins=50,
    output_path=None,
    filename="corner_grid.pdf",
):
    """
    Plot an n x n grid of corner plots for pairs (Ωm, w).
    Each cell shows: joint posterior (contours at 68/95/99.7%) and marginal posteriors,
    with true lines and truth marker overlaid.

    Styling tweaks:
      - Contours and histograms in steelblue
      - Reduced label size and tick lengths
      - Show tick labels for all plots
      - Histogram axes: ticks only on the bottom (top ticks off)
    """
    N = len(flat_samples_list)
    if N == 0:
        raise ValueError("flat_samples_list is empty; nothing to plot.")

    import math
    if n is None:
        n = int(math.ceil(math.sqrt(N)))
    n_total = min(N, n * n)

    # Outer grid
    fig = plt.figure(figsize=(n * figsize_per_cell, n * figsize_per_cell))
    outer = fig.add_gridspec(n, n, wspace=0.3, hspace=0.3)

    for k in range(n_total):
        r, cidx = divmod(k, n)
        subfig = fig.add_subfigure(outer[r, cidx])

        # Columns [Ωm, w] as in your reference (flip from [w, Ωm] if needed)
        data = np.flip(flat_samples_list[k], axis=1)

        # Draw the corner on this subfigure with steelblue color and smaller labels
        fig_corner = corner.corner(
            data,
            fig=subfig,
            labels=[r"$\Omega_{m}$", r"$w$"],
            color=color,
            hist_kwargs={"density": True},
            contour_kwargs={"colors": color, "linewidths": 1.5},
            label_kwargs={"fontsize": 11},            # reduced label size
            alpha=0.1,
            plot_contours=True,
            levels=[0.682, 0.95, 0.997],
            plot_datapoints=False,
            fill_contours=False,
            smooth=None,
        )

        # 2x2 axes (Ωm marginal |  , joint;  , w marginal)
        axes = np.array(fig_corner.axes).reshape((2, 2))

        # Overlay true values
        w_true = float(true_w_list[k])
        om_true = float(true_om_list[k])

        # w marginal (bottom-right)
        axes[1, 1].axvline(w_true, linestyle="--", color="k", lw=1.5)
        # Ωm marginal (top-left)
        axes[0, 0].axvline(om_true, linestyle="--", color="k", lw=1.5)
        # Joint (bottom-left)
        axes[1, 0].axvline(om_true, linestyle="--", color="k", lw=1.2)
        axes[1, 0].axhline(w_true, linestyle="--", color="k", lw=1.2)
        axes[1, 0].scatter(om_true, w_true, color="k", s=20, marker="s", zorder=5)

        for ax in fig_corner.get_axes():
            ax.tick_params(axis="both", which="both", labelsize=10, length=3)

        axes[0, 0].tick_params(axis="x", top=False, bottom=True)  # Ωm marginal
        axes[1, 1].tick_params(axis="x", top=False, bottom=True)  # w marginal

        axes[0, 0].tick_params(labelbottom=True, labelleft=True)
        axes[1, 0].tick_params(labelbottom=True, labelleft=True)
        axes[1, 1].tick_params(labelbottom=True, labelleft=True)

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    return fig

def reduced_chisquare(observed, expected, std):
    chi2 = np.sum(((observed - expected) / std) ** 2)
    return chi2 / (len(observed) - 1)

def plot_true_vs_mean_std_func(true_val, mean_val, std_val, xlabel, y1label, y2label, rchisq, output_path):
    """
    All the inputs are list of w and om values
    """
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), 
                         gridspec_kw={'height_ratios':[0.7,0.3]}, sharex=False)
    (ax_w_top, ax_om_top), (ax_w_bot, ax_om_bot) = axes
    # --- w subplot (left column) ---
    # top-left: true vs pred
    ax_w_top.plot([true_val[0].min(), true_val[0].max()],
                  [true_val[0].min(), true_val[0].max()], 'k--', linewidth=1)
    ax_w_top.errorbar(true_val[0], mean_val[0], yerr=std_val[0],fmt='o', zorder=2, markersize=3.0, linewidth=2.0, mec='k', mfc='k', ecolor='steelblue')
    ax_w_top.set_ylabel(y1label[0])
    ax_w_top.text(0.05, 0.95, f'$\\chi^2_r$={rchisq[0]:.2f}',transform=ax_w_top.transAxes, va='top')
    # bottom-left: (true−pred)/true
    ax_w_bot.axhline(0, color='k', linestyle='--', linewidth=1)
    ax_w_bot.errorbar(true_val[0], (true_val[0] - mean_val[0])/true_val[0],
                        yerr=std_val[0]/np.abs(true_val[0]), fmt='o', zorder=2, markersize=3.0, linewidth=2.0, mec='k', mfc='k', ecolor='steelblue')
    ax_w_bot.set_ylabel(y2label[0])
    ax_w_bot.set_xlabel(xlabel[0])
    # --- Ωm subplot (right column) ---
    # top-right: true vs pred
    ax_om_top.plot([true_val[1].min(), true_val[1].max()],
                   [true_val[1].min(), true_val[1].max()], 'k--', linewidth=1)
    ax_om_top.errorbar(true_val[1], mean_val[1], yerr=std_val[1],
                       fmt='o', zorder=2, markersize=3.0, linewidth=2.0, mec='k', mfc='k', ecolor='steelblue')
    ax_om_top.set_ylabel(y1label[1])
    ax_om_top.text(0.05, 0.95, f'$\\chi^2_r$={rchisq[1]:.2f}',
                   transform=ax_om_top.transAxes, va='top')
    # bottom-right: (true−pred)/true
    ax_om_bot.axhline(0, color='k', linestyle='--', linewidth=1)
    ax_om_bot.errorbar(true_val[1], (true_val[1] - mean_val[1])/true_val[1],
                       yerr=std_val[1]/np.abs(true_val[1]),
                       fmt='o', zorder=2, markersize=3.0, linewidth=2.0, mec='k', mfc='k', ecolor='steelblue')
    ax_om_bot.set_ylabel(y2label[1])
    ax_om_bot.set_xlabel(xlabel[1])
    plt.tight_layout()
    plt.savefig(output_path+"parity_mcmc.pdf", dpi=300)
    return 

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process some parameters.')

    parser.add_argument('--model_path', type=str, required=True, help='Trained model to be used')
    parser.add_argument('--model_name', type=str, required=True, help='Trained model to be used')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output')
    parser.add_argument('--w_min', type=float, default=-2.0, help='Minimum value of w')
    parser.add_argument('--w_max', type=float, default=0.0, help='Maximum value of w')
    parser.add_argument('--om_min', type=float, default=0.1, help='Minimum value of Omega_m')
    parser.add_argument('--om_max', type=float, default=0.5, help='Maximum value of Omega_m')
    parser.add_argument('--n_lenses', type=int, default=6, help='Number of lenses to consider')
    
    # Parse the arguments
    args = parser.parse_args()

    posterior_path = f'/deepskieslab/stronglensing/hsbi/nre_posterior/{args.model_name}/'

    if not os.path.exists(posterior_path):
        os.mkdir(posterior_path)

    if not os.path.exists(posterior_path+f'{args.data_dir}_mcmc.npz'):

        w_min = args.w_min
        w_max = args.w_max
        om_min = args.om_min
        om_max = args.om_max    
        n_pairs = args.n_lenses
    
        start_time = time.time()
    
        
        w_column_name = "w0-g"
        om_column_name = 'Om0-g'
        zl_column_name = 'PLANE_1-REDSHIFT-g'
        zs_column_name = 'PLANE_2-REDSHIFT-g'
        v_column_name = 'PLANE_1-OBJECT_1-MASS_PROFILE_1-sigma_v-g'
    
        flat_samples_list = []
        samples_w_list = []
        samples_om_list = []
        true_w_list = []
        true_om_list = []
    
        for i in range(0,n_pairs):
            try:
                # data_name = f'data_{i}'
                # image_dir = args.data_dir + data_name
                image_dir = f'/deepskieslab/stronglensing/hsbi/datasets/test_data/{args.data_dir}/data_{i}'
                print('Processing (w,om) for 100 images: pair', i+1)
                print('Image directory: ', image_dir)
                posterior_coverage = PosteriorCoverage(args.model_path, args.model_name, 'fake_path', w_column_name, om_column_name, zl_column_name, zs_column_name, v_column_name, w_min, w_max, om_min, om_max)
                flat_samples, samples_w, samples_om, true_w, true_om = posterior_coverage.get_samples_wom(image_dir, walkers=4, nsteps=500, nburn=100)
    
                flat_samples_list.append(flat_samples)
                samples_w_list.append(samples_w)
                samples_om_list.append(samples_om)
                true_w_list.append(true_w)
                true_om_list.append(true_om)
                
                # Free up memory
                del posterior_coverage
                del flat_samples
                del true_w
                del true_om
                tf.keras.backend.clear_session()
                gc.collect()
        
            except:
                continue
       
        flat_samples_list = np.array(flat_samples_list)
        samples_w_list = np.array(samples_w_list)
        samples_om_list = np.array(samples_om_list)
        true_w_list = np.array(true_w_list)
        true_om_list = np.array(true_om_list)

        print(true_w_list, true_om_list)
        
    
        # Save the samples
        np.savez(posterior_path+f'{args.data_dir}_mcmc.npz', flat_samples_list = flat_samples_list,
                 samples_w_list = samples_w_list, samples_om_list = samples_om_list, 
                 true_w_list = true_w_list, true_om_list = true_om_list)
        
        end_time = time.time()
    
        print('Time taken for '+str(n_pairs)+' pairs of 100 images:', end_time - start_time)

    else:
        print("Loading the mcmc posterior data from {}".format(posterior_path))
        posterior_data = np.load(posterior_path+f'{args.data_dir}_mcmc.npz')
        flat_samples_list = posterior_data['flat_samples_list']
        samples_w_list = posterior_data['samples_w_list']
        samples_om_list = posterior_data['samples_om_list']
        true_w_list = posterior_data['true_w_list']
        true_om_list = posterior_data['true_om_list']

    w_mean_list = []
    w_std_list = []
    om_mean_list = []
    om_std_list = []
    
    
    for i in range(0, samples_w_list.shape[0]):
        w_mean = np.mean(samples_w_list[i])
        w_std = np.std(samples_w_list[i])
        om_mean = np.mean(samples_om_list[i])
        om_std = np.std(samples_om_list[i])
        w_mean_list.append(w_mean)
        w_std_list.append(w_std)
        om_mean_list.append(om_mean)
        om_std_list.append(om_std)
        del w_mean, w_std, om_mean, om_std
    
    w_mean_list = np.array(w_mean_list)
    w_std_list = np.array(w_std_list)
    om_mean_list = np.array(om_mean_list)
    om_std_list = np.array(om_std_list)
    
    
    print("Calculate the reduced chi square of the fit...")
    
    reduced_chisquare_w = reduced_chisquare(true_w_list, w_mean_list, w_std_list)
    reduced_chisquare_om = reduced_chisquare(true_om_list, om_mean_list, om_std_list)
    print(f"Reduced chisquare for w: {reduced_chisquare_w:.2f}")
    print(f"Reduced chisquare for om: {reduced_chisquare_om:.2f}")

    print("Plotting true vs mean and std...")
    plot_true_vs_mean_std_func(
        [true_w_list, true_om_list],
        [w_mean_list, om_mean_list],
        [w_std_list, om_std_list],
        [r'$w$', r'$\Omega_m$'],
        [r'$w$ True', r'$\Omega_m$ True'],
        [r'$(w - w_{True})/w_{True}$', r'$(\Omega_m - \Omega_{m,True})/\Omega_{m,True}$'],
        [reduced_chisquare_w, reduced_chisquare_om],
        f"{args.output_path}")

    print("Calculating the coverage fraction...")
    percentile_array = np.linspace(0,100,21)
    n_pairs = len(true_w_list)
    coverage_fraction_w = calculate_coverage_fraction_density_estimator(n_pairs, samples_w_list, 
                                                                    true_w_list, percentile_array)
    coverage_fraction_om = calculate_coverage_fraction_density_estimator(n_pairs, samples_om_list, true_om_list, 
                                                                     percentile_array)

    print("Plotting the coverage fraction...")
    percentile_array_norm = np.array(percentile_array)/100

    default_cycler = (cycler(color='bgrcmyk') *
                        cycler(linestyle=['-', '-.']))

    fig, ax = plt.subplots(1,1,figsize=(8, 8))
    plt.plot(percentile_array_norm, coverage_fraction_w, marker='o', markersize=3, color='steelblue', label=r'$w$')
    plt.plot(percentile_array_norm, coverage_fraction_om, marker='o', markersize=3, color='orange', label=r'$\Omega_m$')
    plt.legend(loc='upper left', fontsize=14)
    plt.plot([0,0.5,1],[0,0.5,1], 'k--', zorder=1000)
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.text(0.03,0.85,'Underconfident',horizontalalignment='left', fontsize=14)
    plt.text(0.7,0.05,'Overconfident',horizontalalignment='right', fontsize=14)
    plt.xlabel('Confidence Interval of the Posterior Volume', fontsize=18, labelpad=20)
    plt.ylabel('Fraction of Lenses within Posterior Volume', fontsize=18, labelpad=20)

    ax.tick_params(axis='both', which='both', labelsize=16)
    plt.tight_layout()
    plt.savefig(f"{args.output_path}coverage_plot_mcmc.pdf", dpi=300)

    # plot the joint posterior with marginals for the first 25 lenses
    print("Plotting the joint posterior with marginals for the first 25 lenses...")
    
    plot_joint_with_marginals(
    flat_samples_list=flat_samples_list,
    samples_w_list=samples_w_list,
    samples_om_list=samples_om_list,
    true_w_list=true_w_list,
    true_om_list=true_om_list,
    n=5,  # 5x5 grid
    output_path=args.output_path,
    filename=f"{args.output_path}joint_posterior_mcmc.pdf",)



if __name__ == '__main__':
    main()
