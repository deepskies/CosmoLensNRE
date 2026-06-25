#!/usr/bin/env python3
"""
Calibration script for trained NRE model using SBC (Simulation-Based Calibration).

This script optimizes alpha and beta parameters to minimize the Earth Mover's Distance
between rank histograms and uniform distribution.
"""

import numpy as np
import pandas as pd
import emcee
import time
import gc
import argparse
from datetime import datetime
import csv
import os
import traceback
import random

import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance
from scipy.stats import qmc  # for Sobol/Latin sampling (installed with scipy)

from concurrent.futures import ThreadPoolExecutor

from dataclasses import dataclass
from typing import Tuple, List, Optional

# use cpu instead of gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Clear any existing TensorFlow state
tf.keras.backend.clear_session()
gc.collect()

rnd_seed = 140

def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    random.seed(seed)
    print(f"Random seed set to {seed}")


def get_logr_mcmc(model, images, aparam, sample_theta):
    '''
    Function to predict the log likelihood-to-evidence ratio (logr) of all the test data at a time

    Input:
    model: The trained model 
    images: test images
    sample_theta: a list of theta values to compute logr for

    Output:
    log r 
    '''
    theta_array = np.array([sample_theta]*images.shape[0]).astype(np.float32)
    images = np.ascontiguousarray(images.astype(np.float32))
    aparam = np.ascontiguousarray(aparam.astype(np.float32))
    theta_array = np.ascontiguousarray(theta_array)

    # Validate shapes before prediction
    if images.shape[0] != aparam.shape[0] or images.shape[0] != theta_array.shape[0]:
        raise ValueError(f"Batch size mismatch: images={images.shape[0]}, aparam={aparam.shape[0]}, theta={theta_array.shape[0]}")
    
    # # Debug prints for troubleshooting
    # print(f"Debug - Images: {images.shape}, Aparam: {aparam.shape}, Theta: {theta_array.shape}")
    
    try:
        output = model.predict([images, aparam, theta_array], verbose=0).flatten()
        return output
    except Exception as e:
        print(f"Model prediction error: {e}")
        print(f"Input dtypes - Images: {images.dtype}, Aparam: {aparam.dtype}, Theta: {theta_array.dtype}")
        raise


def log_prior(theta, w_low=-1.5, w_high=-0.5, om_low=0.1, om_high=0.5):
    """
    prior for w, om
    """
    w, om = theta
    if w_low < w < w_high and om_low < om < om_high:
        return 0.0
    return -np.inf


def log_likelihood(theta, data, aparam, w_low, w_high, om_low, om_high, model, alpha, beta):
    """
    Calculate the log likelihood + log prior
    """
    lp = log_prior(theta, w_low, w_high, om_low, om_high)
    if not np.isfinite(lp):
        return -np.inf
    logr_array = get_logr_mcmc(model, data, aparam, theta)
    # transform sum logr to alpha*sum(logr) + beta
    ll = alpha*np.sum(logr_array) + beta
    return ll+lp


def get_posterior_mcmc(data, aparam, w_low, w_high, om_low, om_high, model, alpha, beta, 
                       walkers=10, nsteps=10000, nburn=200, initial_w=-1.0, initial_om=0.3, 
                       multithread=False):
    """
    MCMC sampling

    Input:
    data: images
    aparam: astrophysics parameters
    w_low, w_high: prior range for w    
    om_low, om_high: prior range for om
    model: trained NRE model
    alpha, beta: parameters to calibrate the logr output

    Output: sampler, samples, flat_samples
    """
    pos = np.array([initial_w, initial_om]) + np.array([initial_w, initial_om])*1e-3* np.random.randn(walkers, 2)
    nwalkers, ndim = pos.shape

    # Clear any existing TensorFlow sessions before starting
    tf.keras.backend.clear_session()
    
    if multithread:
        if multithread:
            with ThreadPoolExecutor(20) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, 
                                              args=(data, aparam, w_low, w_high, om_low, om_high, model, alpha, beta), 
                                              pool=pool)
                pos, lp, _ = sampler.run_mcmc(pos, nsteps, progress=False)
            
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, 
                                      args=(data, aparam, w_low, w_high, om_low, om_high, model, alpha, beta))
        pos, lp, _ = sampler.run_mcmc(pos, nsteps, progress=False)

    samples = sampler.get_chain(discard=int(nsteps/4), flat=False)
    flat_samples = sampler.get_chain(discard=int(nsteps/4), flat=True)

    return sampler, samples, flat_samples



@dataclass
class MCMCConfig:
    walkers: int = 2
    nsteps: int = 80
    nburn: int = 20
    n_subset: int = 20
    multithread: bool = False  

# class to optimize alpha and beta parameters
class AlphaBetaOptimizer:
    def __init__(self, num_sbc_runs=10, model_path=None, data_path=None, model_name=None):
        self.num_sbc_runs = num_sbc_runs
        self.iteration_count = 0 
        self.optimization_history = []  # Store all iterations
        self.log_file = None
        
        # Model and data paths
        self.model_path = model_path or '/deepskieslab/stronglensing/hsbi/models/'
        self.data_path = data_path or '/deepskieslab/stronglensing/hsbi/datasets/calibrate_sbc_data/entire_region/'
        self.model_name = model_name or 'entire_region_narrowprior.keras'
        
        # Load model
        print(f"Loading model from: {self.model_path + self.model_name}")
        self.model = tf.keras.models.load_model(self.model_path + self.model_name)
        
        # Column names
        self.w_column_name = "w0-g"  # Dark energy equation-of-state parameter 
        self.om_column_name = 'Om0-g'
        self.zl_column_name = 'PLANE_1-REDSHIFT-g'
        self.zs_column_name = 'PLANE_2-REDSHIFT-g'
        self.v_column_name = 'PLANE_1-OBJECT_1-MASS_PROFILE_1-sigma_v-g'
        
        # Parameter ranges for MCMC sampling
        self.w_min = -2.5
        self.w_max = -0.34
        self.om_min = 0.0
        self.om_max = 1.2

    def _initialize_log_file(self):
        """Initialize CSV file for logging optimization history"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"optimization_log_{timestamp}.csv"
        
        # Create CSV file with headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'alpha', 'beta', 'loss', 'w_loss', 'om_loss', 'timestamp', 'time_taken_sec'])
        
        print(f"📝 Logging optimization to: {self.log_file}")

    def _log_iteration(self, alpha, beta, loss, w_loss=None, om_loss=None, time_taken=None):
        """Log a single iteration to file"""
        if self.log_file is None:
            self._initialize_log_file()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Append to CSV file
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.iteration_count, 
                f"{alpha:.6f}", 
                f"{beta:.6f}", 
                f"{loss:.6f}",
                f"{w_loss:.6f}" if w_loss is not None else "N/A",
                f"{om_loss:.6f}" if om_loss is not None else "N/A",
                timestamp,
                f"{time_taken:.2f}" if time_taken is not None else "N/A"
            ])
        
        # Also store in memory
        self.optimization_history.append({
            'iteration': self.iteration_count,
            'alpha': alpha,
            'beta': beta,
            'loss': loss,
            'w_loss': w_loss,
            'om_loss': om_loss,
            'timestamp': timestamp,
            'time_taken': time_taken
        })

    # Small utility to cache expensive EMD evaluations across runs
    def _init_cache(self):
        if not hasattr(self, "_loss_cache"):
            self._loss_cache = {}  # key: (alpha, beta, walkers, nsteps, nburn, n_subset)

    def _cache_key(self, alpha: float, beta: float, cfg: MCMCConfig) -> Tuple:
        # round to avoid float noise in dict keys
        return (round(alpha, 8), round(beta, 8), cfg.walkers, cfg.nsteps, cfg.nburn, cfg.n_subset)


    def _loss_function_binned(self, rank_values):
        """
        Parameters:
        rank_values: array of rank statistics from SBC
        """
        # n_bins = max(1, int(self.num_sbc_runs/20))  # Ensure at least 1 bin
        n_bins = int(6)  # Ensure at least 1 bin

        binned_ranks, _ = np.histogram(rank_values, bins=n_bins)
        binned_ranks = binned_ranks / np.sum(binned_ranks)

        # print("Binned ranks", binned_ranks)

        binned_uniform = np.ones(len(binned_ranks)) / len(binned_ranks)

        bins = np.arange(len(binned_ranks)+1)

        # calculate emd between the binned rank histogram and binned uniform distribution
        emd = wasserstein_distance(bins[:-1], bins[:-1], binned_ranks, binned_uniform)

        # # plot and save the rank histogram and uniform distribution for visualization
        # plt.figure(figsize=(8, 5))
        # plt.step(bins[:-1], binned_ranks, where='mid', alpha=0.7, label='Rank Histogram', color='blue')
        # plt.step(bins[:-1], binned_uniform, where='mid', alpha=0.7, label='Uniform Distribution', color='orange')
        # plt.xlabel('Bins')
        # plt.ylabel('Frequency')
        # plt.title(f'Rank Histogram - {param_name}\nα={alpha:.3f}, β={beta:.3f}, Loss={loss:.3f}')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f'rank_histogram_iteration_{param_name}_{self.iteration_count}.png')
        # plt.close()


        return emd

    def _plot_rank_histogram(self, rank_values, alpha, beta, loss, param_name):
        """
        Plot and save the rank histogram and uniform distribution for visualization
        
        Parameters:
        rank_values: array of rank statistics
        alpha: current alpha parameter
        beta: current beta parameter
        loss: current loss value
        param_name: parameter name (e.g., 'w' or 'om')
        """
        n_bins = int(6)
        binned_ranks, _ = np.histogram(rank_values, bins=n_bins)
        binned_ranks = binned_ranks / np.sum(binned_ranks)
        binned_uniform = np.ones(len(binned_ranks)) / len(binned_ranks)
        bins = np.arange(len(binned_ranks)+1)
        
        plt.figure(figsize=(8, 5))
        plt.step(bins[:-1], binned_ranks, where='mid', alpha=0.7, label='Rank Histogram', color='blue')
        plt.step(bins[:-1], binned_uniform, where='mid', alpha=0.7, label='Uniform Distribution', color='orange')
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.title(f'Rank Histogram - {param_name}\nα={alpha:.4f}, β={beta:.4f}, Loss={loss:.6f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'rank_histogram_{param_name}_iter_{self.iteration_count}.png')
        plt.close()

    def evaluate_loss(self, alpha: float, beta: float, cfg: MCMCConfig) -> float:
        """
        Compute EMD loss for a given (alpha, beta) at a specified MCMC fidelity.
        Same logic as _compute_loss_for_params, but parameterized by cfg and with caching.
        """
        self._init_cache()
        cache_key = self._cache_key(alpha, beta, cfg)
        if cache_key in self._loss_cache:
            return self._loss_cache[cache_key]

        iteration_start_time = time.time()
        rank_hist_w_temp, rank_hist_om_temp = [], []

        # Bounds check (use your narrower/stricter ranges here if desired)
        if alpha <= 0 or alpha >= 2.0 or beta <= -5.0 or beta >= 5.0:
            loss = 1000.0
            self._loss_cache[cache_key] = loss
            return loss

        for n in range(0, self.num_sbc_runs):
            try:
                tf.keras.backend.clear_session()
                gc.collect()

                # print(f"Processing cosmology {n}")

                image_dir = f'data_{n}'
                images = np.load(self.data_path + image_dir + '/CONFIGURATION_1_images.npy', allow_pickle=True)
                metadata = pd.read_csv(self.data_path + image_dir + '/CONFIGURATION_1_metadata.csv')

                images = np.einsum('lkij->lijk', images)
                aparms = metadata[[self.zl_column_name, self.zs_column_name, self.v_column_name]].to_numpy().astype(np.float32)

                edge_size = images.shape[1]
                # image_sums = np.sum(images, axis=(1, 2), keepdims=True)
                # image_sums = np.where(image_sums == 0, 1.0, image_sums)
                images = edge_size * edge_size * (images / np.sum(images, axis=(1, 2), keepdims=True))
                images = images.reshape(images.shape[0], edge_size, edge_size, 1)

                true_w = metadata[self.w_column_name].unique()[0]
                true_om = metadata[self.om_column_name].unique()[0]

                # Subset per fidelity
                n_subset = min(cfg.n_subset, images.shape[0], aparms.shape[0])
                if n_subset <= 0:
                    continue
                images_subset = np.ascontiguousarray(images[:n_subset])
                aparms_subset = np.ascontiguousarray(aparms[:n_subset])

                sampler, samples, flat_samples = get_posterior_mcmc(
                    images_subset, aparms_subset,
                    self.w_min, self.w_max, self.om_min, self.om_max,
                    self.model, alpha, beta,
                    walkers=cfg.walkers, nsteps=cfg.nsteps, nburn=cfg.nburn,
                    initial_w=-1.5, initial_om=0.31,
                    multithread=cfg.multithread
                )

                if flat_samples.size == 0:
                    continue
                w_samples = flat_samples[:, 0]
                om_samples = flat_samples[:, 1]

                w_rank = np.sum(w_samples < true_w)
                om_rank = np.sum(om_samples < true_om)

                # print("w rank: ", w_rank)
                # print("om rank: ", om_rank)
                rank_hist_w_temp.append(w_rank)
                rank_hist_om_temp.append(om_rank)

                # cleanup
                del images, metadata, aparms, images_subset, aparms_subset
                del sampler, samples, flat_samples, w_samples, om_samples
                gc.collect()
                tf.keras.backend.clear_session()

            except Exception as e:
                # Penalize failures to keep search moving
                loss = 100.0
                self._loss_cache[cache_key] = loss
                return loss

        if len(rank_hist_w_temp) == 0 or len(rank_hist_om_temp) == 0:
            loss = 100.0
            self._loss_cache[cache_key] = loss
            return loss

        loss_w = self._loss_function_binned(np.array(rank_hist_w_temp))
        loss_om = self._loss_function_binned(np.array(rank_hist_om_temp))

        self._plot_rank_histogram(np.array(rank_hist_w_temp), alpha, beta, loss_w, 'w')
        self._plot_rank_histogram(np.array(rank_hist_om_temp), alpha, beta, loss_om, 'om')

        combined_loss = (loss_w + loss_om) / 2
        result = (combined_loss, loss_w, loss_om)
        self._loss_cache[cache_key] = result
        return result

    def _sample_points(self, alpha_bounds: Tuple[float, float], beta_bounds: Tuple[float, float],
                       n_points: int, method: str = "sobol") -> np.ndarray:
        """
        Sample points in 2D box using Sobol (default), Latin hypercube, or linear grid.
        Returns array of shape (n_points, 2) in [alpha, beta].
        """
        a_min, a_max = alpha_bounds
        b_min, b_max = beta_bounds
        if method == "sobol":
            engine = qmc.Sobol(d=2, scramble=True, seed=rnd_seed)
            X = engine.random(n_points)
        elif method == "latin":
            engine = qmc.LatinHypercube(d=2, seed=rnd_seed)
            X = engine.random(n_points)
        elif method == "linear-grid":
            # Closest to a "linear scheduler": uniform linear sweeps
            n_side = int(np.ceil(np.sqrt(n_points)))
            alpha_lin = np.linspace(a_min, a_max, n_side)
            beta_lin = np.linspace(b_min, b_max, n_side)
            A, B = np.meshgrid(alpha_lin, beta_lin, indexing="xy")
            X = np.stack([A.ravel(), B.ravel()], axis=1)
            if X.shape[0] > n_points:
                X = X[:n_points]
            return X
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        # scale to bounds
        X[:, 0] = a_min + X[:, 0] * (a_max - a_min)
        X[:, 1] = b_min + X[:, 1] * (b_max - b_min)
        return X

    def _shrink_box(self, center: Tuple[float, float],
                    box: Tuple[Tuple[float, float], Tuple[float, float]],
                    factor: float = 0.5) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Shrink a 2D box around a center by 'factor' in each dimension.
        """
        (a_min, a_max), (b_min, b_max) = box
        ca, cb = center
        a_half = max((a_max - a_min) * factor / 2.0, 1e-6)
        b_half = max((b_max - b_min) * factor / 2.0, 1e-6)
        new_a = (max(a_min, ca - a_half), min(a_max, ca + a_half))
        new_b = (max(b_min, cb - b_half), min(b_max, cb + b_half))
        return (new_a, new_b)

    def scheduled_search(
        self,
        alpha_bounds: Tuple[float, float] = (0.05, 2.0),
        beta_bounds: Tuple[float, float] = (-5.0, 5.0),
        rounds: int = 3,
        initial_points: int = 20,
        top_k: int = 5,
        method: str = "sobol",
        refine_factor: float = 0.4,
        loss_threshold: float = 0.4,
        max_round1_repeats: int = 2,
        min_good_points: int = 3,  # NEW: require at least this many points below threshold
        fidelity_schedule: Optional[List[MCMCConfig]] = None,
        per_round_points: Optional[List[int]] = None,):
        """
        Multi-round exploration of (alpha, beta) with increasing fidelity.
        - Round 1: wide exploration, cheap fidelity
        - Check: if < min_good_points have loss <= loss_threshold, repeat round 1
        - Subsequent rounds: shrink around top-k, add points, increase fidelity
        
        Parameters:
        loss_threshold: loss threshold for considering a point "good"
        min_good_points: minimum number of points below threshold needed to proceed
        max_round1_repeats: maximum times to repeat round 1 before proceeding anyway
        """
        self._initialize_log_file()
        self._init_cache()
    
        if fidelity_schedule is None:
            fidelity_schedule = [
                MCMCConfig(walkers=4, nsteps=100, nburn=10, n_subset=20),   # very cheap
                MCMCConfig(walkers=4, nsteps=120, nburn=20, n_subset=50), # medium
                MCMCConfig(walkers=4, nsteps=200, nburn=30, n_subset=100), # higher fidelity
            ]
        if per_round_points is None:
            per_round_points = [initial_points, max(10, initial_points // 2), max(6, initial_points // 3)]
    
        rounds = min(rounds, len(fidelity_schedule))
        alpha_box = alpha_bounds
        beta_box = beta_bounds
    
        all_results = []  # for a summary
        current_centers = None  # centers to refine
        
        # Round 1: repeated exploration
        r = 0
        cfg = fidelity_schedule[r]
        round1_repeats = 0
        best_loss_round1 = float('inf')
        
        print(f"\n{'='*80}")
        print(f"ROUND 1: INITIAL EXPLORATION (cheap fidelity)")
        print(f"{'='*80}")
        print(f"Loss threshold: {loss_threshold}")
        print(f"Min good points required: {min_good_points}")
        print(f"Max round 1 repeats: {max_round1_repeats}")
        
        while round1_repeats <= max_round1_repeats:
            n_points = per_round_points[r] + (round1_repeats * 5)  # Add more points on repeats
            print(f"\n--- Round 1, Attempt {round1_repeats + 1} ---")
            print(f"Config: {cfg} | Points: {n_points} | Box: {alpha_box}, {beta_box}")
            
            # Sample points in current box
            if round1_repeats == 0:
                candidates = self._sample_points(alpha_box, beta_box, n_points, method)
            else:
                # On repeats, sample around previous best center
                pts_per_center = max(1, n_points // len(current_centers))
                candidate_list = []
                for ca, cb in current_centers:
                    (a_min, a_max), (b_min, b_max) = self._shrink_box(
                        (ca, cb), (alpha_box, beta_box), factor=refine_factor
                    )
                    candidate_list.append(
                        self._sample_points((a_min, a_max), (b_min, b_max), pts_per_center, method)
                    )
                candidates = np.vstack(candidate_list)
            
            # Evaluate all candidates
            round_results = []
            for (a, b) in candidates:
                t0 = time.time()
                combined_loss, w_loss, om_loss = self.evaluate_loss(float(a), float(b), cfg)
                dt = time.time() - t0
                self.iteration_count += 1
                self._log_iteration(float(a), float(b), combined_loss, w_loss=w_loss, om_loss=om_loss, time_taken=dt)
                round_results.append((float(a), float(b), combined_loss))
            
            round_df = pd.DataFrame(round_results, columns=["alpha", "beta", "loss"])
            all_results.append(round_df)
            
            # Select top-k as centers for refinement
            round_df = round_df.sort_values("loss", ascending=True).reset_index(drop=True)
            current_centers = round_df.loc[:max(0, top_k - 1), ["alpha", "beta"]].to_numpy()
            
            # Get best loss from this attempt
            best_a, best_b, best_loss_round1 = round_df.loc[0, ["alpha", "beta", "loss"]]
            
            # Count how many points are below threshold
            num_good_points = (round_df['loss'] <= loss_threshold).sum()
            
            print(f"Round 1 Attempt {round1_repeats + 1} best: alpha={best_a:.6f} beta={best_b:.6f} loss={best_loss_round1:.6f}")
            print(f"  Good points (loss <= {loss_threshold}): {num_good_points}/{len(round_df)}")
            
            # Check if we have enough good points
            if num_good_points >= min_good_points:
                print(f"\n✅ Found {num_good_points} good points >= threshold {min_good_points} - Moving to Round 2")
                
                # Shrink global box around the best point found
                (alpha_box, beta_box) = self._shrink_box(
                    (best_a, best_b), (alpha_bounds, beta_bounds), factor=refine_factor
                )
                break
            elif round1_repeats < max_round1_repeats:
                print(f"\n⚠️  Only {num_good_points} good points (need {min_good_points}) - Repeating Round 1")
                round1_repeats += 1
            else:
                print(f"\n⚠️  Only {num_good_points} good points (need {min_good_points}) - Max repeats reached, moving to Round 2 anyway")
                (alpha_box, beta_box) = self._shrink_box(
                    (best_a, best_b), (alpha_bounds, beta_bounds), factor=refine_factor
                )
                break
        
        # Rounds 2+ with increased fidelity
        for r in range(1, rounds):
            cfg = fidelity_schedule[r]
            n_points = per_round_points[r]
            print(f"\n{'='*80}")
            print(f"ROUND {r+1}: REFINEMENT (higher fidelity)")
            print(f"{'='*80}")
            print(f"Config: {cfg} | Points: {n_points} | Box: {alpha_box}, {beta_box}")
            
            # Sample around top-k centers from previous round
            pts_per_center = max(1, n_points // len(current_centers))
            candidate_list = []
            for ca, cb in current_centers:
                (a_min, a_max), (b_min, b_max) = self._shrink_box(
                    (ca, cb), (alpha_box, beta_box), factor=refine_factor
                )
                candidate_list.append(
                    self._sample_points((a_min, a_max), (b_min, b_max), pts_per_center, method)
                )
            candidates = np.vstack(candidate_list)
            
            # Evaluate
            round_results = []
            for (a, b) in candidates:
                t0 = time.time()
                combined_loss, w_loss, om_loss = self.evaluate_loss(float(a), float(b), cfg)
                dt = time.time() - t0
                self.iteration_count += 1
                self._log_iteration(float(a), float(b), combined_loss, w_loss=w_loss, om_loss=om_loss, time_taken=dt)
                round_results.append((float(a), float(b), combined_loss))
    
            round_df = pd.DataFrame(round_results, columns=["alpha", "beta", "loss"])
            all_results.append(round_df)
    
            # Select top-k as centers for next round
            round_df = round_df.sort_values("loss", ascending=True).reset_index(drop=True)
            current_centers = round_df.loc[:max(0, top_k - 1), ["alpha", "beta"]].to_numpy()
    
            # Shrink global box around the very best to drive later focus
            best_a, best_b, best_loss = round_df.loc[0, ["alpha", "beta", "loss"]]
            (alpha_box, beta_box) = self._shrink_box((best_a, best_b), (alpha_bounds, beta_bounds), factor=refine_factor)
    
            print(f"Round {r+1} best: alpha={best_a:.6f} beta={best_b:.6f} loss={best_loss:.6f}")
    
        # Final pick across all rounds
        results_df = pd.concat(all_results, ignore_index=True)
        best_idx = results_df["loss"].idxmin()
        best_alpha = float(results_df.loc[best_idx, "alpha"])
        best_beta = float(results_df.loc[best_idx, "beta"])
        best_loss = float(results_df.loc[best_idx, "loss"])
    
        print(f"\n{'='*80}")
        print(f"🎯 SCHEDULED SEARCH COMPLETED")
        print(f"{'='*80}")
        print(f"Best: alpha={best_alpha:.6f} beta={best_beta:.6f} loss={best_loss:.6f}")
    
        # Persist a grid-results CSV alongside the iteration log
        if self.log_file:
            grid_file = self.log_file.replace(".csv", "_scheduled_search.csv")
            results_df.to_csv(grid_file, index=False)
            print(f"Saved all results to: {grid_file}")

        return best_alpha, best_beta, best_loss


def main():
    set_seed(rnd_seed)
    
    parser = argparse.ArgumentParser(description='Calibrate NRE model using SBC')
    parser.add_argument('--num_sbc_runs', type=int, default=20, 
                        help='Number of SBC runs for optimization (default: 20)')
    parser.add_argument('--model_path', type=str, 
                        default='/deepskieslab/stronglensing/hsbi/models/',
                        help='Path to model directory')
    parser.add_argument('--data_path', type=str, 
                        default='/deepskieslab/stronglensing/hsbi/datasets/calibrate_sbc_data/entire_region/',
                        help='Path to data directory')
    parser.add_argument('--model_name', type=str, 
                        default='entire_region_narrowprior.keras',
                        help='Model filename')
    parser.add_argument('--search', type=str, default='scheduled',
                        choices=['scipy', 'scheduled'],
                        help='Use local scipy minimize or global scheduled search')
    parser.add_argument('--alpha_bounds', type=float, nargs=2, default=[0.05, 2.0],
                        help='Alpha bounds [min, max] for scheduled search')
    parser.add_argument('--beta_bounds', type=float, nargs=2, default=[-5.0, 5.0],
                        help='Beta bounds [min, max] for scheduled search')
    parser.add_argument('--rounds', type=int, default=3, help='Rounds for scheduled search')
    parser.add_argument('--initial_points', type=int, default=20, help='Points in round 1')
    parser.add_argument('--top_k', type=int, default=5, help='Top centers to refine per round')
    parser.add_argument('--sampling', type=str, default='sobol',
                        choices=['sobol', 'latin', 'linear-grid'],
                        help='Sampler for scheduled search')
    parser.add_argument('--refine_factor', type=float, default=0.4,
                        help='Box shrink factor per round (0<factor<1)')
    parser.add_argument('--loss_threshold', type=float, default=0.3,
                        help='Loss threshold for considering a point "good" (default: 0.4)')
    parser.add_argument('--min_good_points', type=int, default=3,
                        help='Minimum number of points below threshold to proceed to Round 2 (default: 3)')
    parser.add_argument('--max_round1_repeats', type=int, default=10,
                        help='Max times to repeat round 1 if not enough good points (default: 2)')

    
    args = parser.parse_args()
    
    print("="*60)
    print("NRE MODEL CALIBRATION USING SBC")
    print("="*60)
    print(f"Number of SBC runs: {args.num_sbc_runs}")
    print(f"Model path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Model name: {args.model_name}")
    # print(f"Initial alpha: {args.initial_alpha}")
    # print(f"Initial beta: {args.initial_beta}")
    # print(f"Optimization method: {args.method}")
    # print(f"Max iterations: {args.max_iterations}")
    print("="*60)
    
    try:
        optimizer = AlphaBetaOptimizer(
            num_sbc_runs=args.num_sbc_runs,
            model_path=args.model_path,
            data_path=args.data_path,
            model_name=args.model_name
        )

        best_alpha, best_beta, best_loss = optimizer.scheduled_search(
            alpha_bounds=tuple(args.alpha_bounds),
            beta_bounds=tuple(args.beta_bounds),
            rounds=args.rounds,
            initial_points=args.initial_points,
            top_k=args.top_k,
            method=args.sampling,
            refine_factor=args.refine_factor,
            loss_threshold=args.loss_threshold,
            min_good_points=args.min_good_points,
            max_round1_repeats=args.max_round1_repeats
        )
        print(f"\n🎉 FINAL (scheduled): alpha={best_alpha:.6f}, beta={best_beta:.6f}, loss={best_loss:.6f}")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
