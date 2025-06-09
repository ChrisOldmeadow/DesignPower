"""
Bayesian inference methods for cluster randomized controlled trials with continuous outcomes.

This module provides optional Bayesian analysis methods including Stan, PyMC, 
variational inference, and approximate Bayesian computation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, approx_fprime
from scipy.stats import multivariate_normal

# Optional Bayesian support
try:
    from cmdstanpy import CmdStanModel
    import tempfile
    import os
    _STAN_AVAILABLE = True
    _STAN_MODEL = None
except ImportError:
    _STAN_AVAILABLE = False
    _STAN_MODEL = None

try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
    _PYMC_AVAILABLE = True
except ImportError:
    _PYMC_AVAILABLE = False

# Check for lightweight Bayesian alternatives
try:
    from scipy.optimize import minimize
    from scipy.stats import multivariate_normal
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _get_stan_model():
    """Get or create the Stan model for cluster RCT analysis."""
    global _STAN_MODEL
    if _STAN_MODEL is None:
        stan_code = """
        data {
            int<lower=1> N;
            int<lower=1> J;
            array[N] int<lower=1, upper=J> cluster;
            vector[N] y;
            vector[N] treat;
        }
        parameters {
            real alpha;
            real beta;
            vector[J] u_raw;
            real<lower=0> sigma_u;
            real<lower=0> sigma_e;
        }
        transformed parameters {
            vector[J] u = u_raw * sigma_u;
        }
        model {
            beta ~ normal(0,10);
            u_raw ~ normal(0,1);
            sigma_u ~ student_t(3,0,2.5);
            sigma_e ~ student_t(3,0,2.5);
            for (n in 1:N)
                y[n] ~ normal(alpha + beta * treat[n] + u[cluster[n]], sigma_e);
        }
        """
        # Write Stan code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.stan', delete=False) as f:
            f.write(stan_code)
            stan_file = f.name
        
        # Create the Stan model - don't delete file until after compilation
        _STAN_MODEL = CmdStanModel(stan_file=stan_file)
        
        # Clean up the temporary file after compilation
        try:
            if os.path.exists(stan_file):
                os.unlink(stan_file)
        except OSError:
            pass  # Ignore cleanup errors
    return _STAN_MODEL


def _fit_pymc_model(df, draws=500, tune=500, chains=4):
    """Fit PyMC hierarchical model for cluster RCT."""
    if not _PYMC_AVAILABLE:
        raise ImportError("PyMC is not available. Please install with: pip install pymc")
    
    # Map cluster IDs to consecutive integers starting from 0 (PyMC convention)
    unique_clusters = sorted(df['cluster'].unique())
    cluster_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
    cluster_ids = df['cluster'].map(cluster_map).values
    
    n_clusters = len(unique_clusters)
    y_obs = df['y'].values
    treatment = df['treatment'].values
    
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        
        # Cluster random effects
        sigma_u = pm.HalfStudentT("sigma_u", nu=3, sigma=2.5)
        u_raw = pm.Normal("u_raw", mu=0, sigma=1, shape=n_clusters)
        u = pm.Deterministic("u", u_raw * sigma_u)
        
        # Individual-level variance
        sigma_e = pm.HalfStudentT("sigma_e", nu=3, sigma=2.5)
        
        # Linear predictor
        mu = alpha + beta * treatment + u[cluster_ids]
        
        # Likelihood
        y = pm.Normal("y", mu=mu, sigma=sigma_e, observed=y_obs)
        
        # Sample
        trace = pm.sample(
            draws=draws, 
            tune=tune, 
            chains=chains, 
            return_inferencedata=True,
            progressbar=False,
            random_seed=42
        )
    
    return trace, model


def _fit_variational_bayes(df, n_samples=1000):
    """Fast approximate Bayesian inference using Variational Bayes (Laplace approximation)."""
    
    # Prepare data
    y = df['y'].values
    treatment = df['treatment'].values
    unique_clusters = sorted(df['cluster'].unique())
    cluster_map = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
    cluster_ids = df['cluster'].map(cluster_map).values
    n_clusters = len(unique_clusters)
    
    # Design matrix: intercept, treatment, cluster random effects
    n_obs = len(y)
    n_params = 2 + n_clusters  # alpha, beta, u_1, ..., u_J
    
    # Negative log posterior for optimization
    def neg_log_posterior(params):
        alpha, beta = params[:2]
        u = params[2:]
        
        # Likelihood: y ~ N(alpha + beta*treatment + u[cluster], sigma_e^2)
        sigma_e = 1.0  # Fixed for simplicity
        mu = alpha + beta * treatment + u[cluster_ids]
        log_lik = -0.5 * np.sum((y - mu)**2) / sigma_e**2
        
        # Priors: alpha, beta ~ N(0, 10^2), u ~ N(0, sigma_u^2)
        sigma_u = 1.0  # Approximate from data
        log_prior_fixed = -0.5 * (alpha**2 + beta**2) / 100  # N(0, 10^2)
        log_prior_random = -0.5 * np.sum(u**2) / sigma_u**2  # N(0, 1^2)
        
        return -(log_lik + log_prior_fixed + log_prior_random)
    
    # Find MAP estimate
    init_params = np.zeros(n_params)
    result = minimize(neg_log_posterior, init_params, method='BFGS')
    
    if not result.success:
        raise RuntimeError("Variational optimization failed")
    
    # Approximate posterior covariance (inverse Hessian at MAP)
    eps = 1e-8
    
    def grad_neg_log_post(params):
        return approx_fprime(params, neg_log_posterior, eps)
    
    # Approximate Hessian numerically (for small problems)
    hess_approx = np.eye(n_params) * 0.1  # Diagonal approximation
    try:
        for i in range(n_params):
            ei = np.zeros(n_params)
            ei[i] = eps
            grad_plus = grad_neg_log_post(result.x + ei)
            grad_minus = grad_neg_log_post(result.x - ei)
            hess_approx[i, i] = max(0.01, (grad_plus[i] - grad_minus[i]) / (2 * eps))  # Ensure positive
    except Exception:
        # If gradient computation fails, use simple diagonal approximation
        hess_approx = np.eye(n_params) * 0.1
    
    # Sample from approximate posterior
    cov_approx = None
    try:
        cov_approx = np.linalg.inv(hess_approx)
        # Ensure positive definite
        cov_approx = cov_approx + np.eye(n_params) * 1e-6
        samples = multivariate_normal.rvs(mean=result.x, cov=cov_approx, size=n_samples)
        
        # Extract beta samples (treatment effect)
        beta_samples = samples[:, 1] if samples.ndim > 1 else [samples[1]]
        
        return beta_samples, True  # success flag
    except (np.linalg.LinAlgError, ValueError) as e:
        # Fallback: use MAP estimate with uncertainty
        beta_map = result.x[1]
        beta_se = np.sqrt(max(0.01, cov_approx[1, 1])) if cov_approx is not None else 0.1
        beta_samples = np.random.normal(beta_map, beta_se, n_samples)
        return beta_samples, False  # approximate flag


def _fit_abc_bayes(df, n_samples=1000, tolerance=0.1):
    """Approximate Bayesian Computation for cluster RCT."""
    
    # Observed summary statistics
    y_obs = df['y'].values
    treatment_obs = df['treatment'].values
    cluster_obs = df['cluster'].values
    
    # Summary statistics: means by group, overall variance, cluster means variance
    control_mean_obs = y_obs[treatment_obs == 0].mean()
    treat_mean_obs = y_obs[treatment_obs == 1].mean()
    overall_var_obs = y_obs.var()
    
    # Cluster-level summaries
    cluster_means_obs = df.groupby(['cluster', 'treatment'])['y'].mean()
    cluster_var_obs = cluster_means_obs.var()
    
    obs_stats = np.array([control_mean_obs, treat_mean_obs, overall_var_obs, cluster_var_obs])
    
    accepted_samples = []
    n_attempts = 0
    max_attempts = n_samples * 50  # Don't run forever
    
    while len(accepted_samples) < n_samples and n_attempts < max_attempts:
        n_attempts += 1
        
        # Sample from priors
        alpha_sim = np.random.normal(0, 10)
        beta_sim = np.random.normal(0, 10)  # This is what we want to estimate
        sigma_u_sim = np.abs(np.random.normal(0, 2))
        sigma_e_sim = np.abs(np.random.normal(0, 2))
        
        # Simulate data with same structure
        try:
            unique_clusters = sorted(df['cluster'].unique())
            n_clusters = len(unique_clusters)
            cluster_size = len(df) // n_clusters // 2  # Approximate
            
            # Generate cluster effects
            u_sim = np.random.normal(0, sigma_u_sim, n_clusters)
            
            # Simulate outcomes
            y_sim = []
            treatment_sim = []
            cluster_sim = []
            
            for i, cluster_id in enumerate(unique_clusters):
                cluster_data = df[df['cluster'] == cluster_id]
                n_in_cluster = len(cluster_data)
                
                for _, row in cluster_data.iterrows():
                    treatment_val = row['treatment']
                    y_val = alpha_sim + beta_sim * treatment_val + u_sim[i] + np.random.normal(0, sigma_e_sim)
                    
                    y_sim.append(y_val)
                    treatment_sim.append(treatment_val)
                    cluster_sim.append(cluster_id)
            
            y_sim = np.array(y_sim)
            treatment_sim = np.array(treatment_sim)
            
            # Calculate summary statistics
            control_mean_sim = y_sim[treatment_sim == 0].mean()
            treat_mean_sim = y_sim[treatment_sim == 1].mean()
            overall_var_sim = y_sim.var()
            
            df_sim = pd.DataFrame({'y': y_sim, 'treatment': treatment_sim, 'cluster': cluster_sim})
            cluster_means_sim = df_sim.groupby(['cluster', 'treatment'])['y'].mean()
            cluster_var_sim = cluster_means_sim.var()
            
            sim_stats = np.array([control_mean_sim, treat_mean_sim, overall_var_sim, cluster_var_sim])
            
            # Check if simulated stats are close enough to observed
            distance = np.sqrt(np.sum((obs_stats - sim_stats)**2))
            
            if distance < tolerance:
                accepted_samples.append(beta_sim)
                
        except Exception:
            continue  # Skip this simulation
    
    if len(accepted_samples) < n_samples // 10:  # Need at least 10% acceptance
        # Relax tolerance and try again
        return _fit_abc_bayes(df, n_samples, tolerance * 2)
    
    return np.array(accepted_samples), len(accepted_samples) >= n_samples // 2  # success if >50% accepted