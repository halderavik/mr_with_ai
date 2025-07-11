import torch
import numpy as np
import pandas as pd
import time  # This was missing
from collections import defaultdict
import matplotlib.pyplot as plt

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

# Configure PyTorch for optimal performance
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type(torch.FloatTensor)

class FastHB_CBC:
    """
    Fast Hierarchical Bayes for CBC data.

    Args:
        X (torch.Tensor): Input tensor of shape (N, T, J, K), where K is the total number of features.
            All attributes, including Price, must be one-hot encoded (dummy variables),
            e.g., columns should include Price_299, Price_399, ..., not a single Price column.
        choice_mask (torch.Tensor): Mask tensor for chosen alternatives.
        num_samples (int): Number of MCMC samples.
        burn_in (int): Number of burn-in iterations.
        device (torch.device, optional): Device to use.
    """
    def __init__(self, X, choice_mask, num_samples=8000, burn_in=2000, device=None):
        # All attributes, including Price, must be one-hot encoded in X (e.g., Price_299, Price_399, ...)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move tensors to device
        self.X = X.to(self.device).float()
        self.choice_mask = choice_mask.to(self.device)
        self.N, self.T, self.J, self.K = X.shape
        
        # MCMC parameters
        self.num_samples = num_samples
        self.burn_in = burn_in
        self.total_iters = burn_in + num_samples
        
        # Prior parameters
        self.nu0 = self.K + 2
        self.V0 = torch.eye(self.K, device=self.device)
        
        # Initialize parameters
        self.beta_i = torch.randn(self.N, self.K, device=self.device) * 0.1
        self.mu = torch.zeros(self.K, device=self.device)
        self.Sigma = torch.eye(self.K, device=self.device)
        self.proposal_std = torch.ones(self.N, device=self.device) * 0.1
        
        # Storage
        self.beta_samples = torch.zeros(num_samples, self.N, self.K, device=self.device)
        self.mu_samples = torch.zeros(num_samples, self.K, device=self.device)
        self.Sigma_samples = torch.zeros(num_samples, self.K, self.K, device=self.device)

    def compute_log_likelihood(self, beta):
        """Batched log-likelihood computation"""
        utilities = torch.einsum('ntjk,nk->ntj', self.X, beta)
        log_probs = utilities - torch.logsumexp(utilities, dim=-1, keepdim=True)
        return (self.choice_mask * log_probs).sum(dim=(-1, -2))

    def update_beta_i(self):
        """Parallel Metropolis-Hastings for all respondents"""
        # Propose new betas
        beta_proposal = self.beta_i + self.proposal_std[:, None] * torch.randn(self.N, self.K, device=self.device)
        
        # Compute log-likelihoods and priors
        ll_current = self.compute_log_likelihood(self.beta_i)
        ll_proposal = self.compute_log_likelihood(beta_proposal)
        
        mvn = torch.distributions.MultivariateNormal(self.mu, self.Sigma)
        log_prior_current = mvn.log_prob(self.beta_i)
        log_prior_proposal = mvn.log_prob(beta_proposal)
        
        # Acceptance ratio
        log_accept = (ll_proposal + log_prior_proposal) - (ll_current + log_prior_current)
        accept = torch.rand(self.N, device=self.device) < torch.exp(torch.clamp(log_accept, max=0.0))
        
        # Update accepted proposals
        self.beta_i = torch.where(accept[:, None], beta_proposal, self.beta_i)
        
        # Adaptive tuning during burn-in
        if self.current_iter < self.burn_in:
            accept_rate = accept.float().mean()
            self.proposal_std *= torch.exp(0.5 * (accept_rate - 0.234))

    def update_population(self):
        """Gibbs sampling for population parameters"""
        # Update mu
        beta_mean = self.beta_i.mean(dim=0)
        mu_cov = self.Sigma / self.N
        self.mu = torch.distributions.MultivariateNormal(beta_mean, mu_cov).sample()
        
        # Update Sigma
        diff = self.beta_i - self.mu
        S = diff.T @ diff
        V_posterior = self.V0 + S
        df_posterior = self.nu0 + self.N
        
        try:
            L_V = torch.linalg.cholesky(V_posterior)
            X0 = torch.randn(df_posterior, self.K, device=self.device)
            YT = torch.linalg.solve_triangular(L_V, X0.T, upper=False)
            A = YT @ YT.T
            self.Sigma = torch.linalg.inv(A)
        except:
            # Fallback if Cholesky fails
            jitter = 1e-3 * torch.eye(self.K, device=self.device)
            self.Sigma = torch.inverse(V_posterior + jitter)

    def run(self):
        """Main MCMC sampling loop"""
        print("\nStarting MCMC sampling...")
        start_time = time.time()
        iter_times = []
        
        for i in range(self.total_iters):
            self.current_iter = i
            iter_start = time.time()
            
            self.update_beta_i()
            self.update_population()
            
            if i >= self.burn_in:
                idx = i - self.burn_in
                self.beta_samples[idx] = self.beta_i
                self.mu_samples[idx] = self.mu
                self.Sigma_samples[idx] = self.Sigma
            
            iter_time = time.time() - iter_start
            iter_times.append(iter_time)
            
            if i % 100 == 0:
                print(f"Iter {i+1}/{self.total_iters} | Time: {iter_time:.4f}s")
        
        total_time = time.time() - start_time
        print(f"\nSampling complete! Total time: {total_time:.2f}s")
        print(f"Average iter time: {np.mean(iter_times):.4f}s")
        print(f"Samples/sec: {self.num_samples / total_time:.1f}")
        
        return self.beta_samples, self.mu_samples, self.Sigma_samples, iter_times

# Load or generate data
#try:
    #X = torch.tensor(np.load('cbc_features_clean.npy')).float().to(device)
    #choices = torch.tensor(np.load('cbc_choices_clean.npy')).long().to(device)
    #print("\nData loaded successfully!")
#except FileNotFoundError:
   # print("\nGenerating new dataset...")
    #from generate_cbc_data import generate_clean_cbc_data
    #X, _, choices = generate_clean_cbc_data()
    #X = torch.tensor(X).float().to(device)
    #choices = torch.tensor(choices).long().to(device)

# Create choice mask
#choice_mask = torch.zeros(X.shape[0], X.shape[1], X.shape[2], device=device)
#choice_mask.scatter_(2, choices.unsqueeze(-1), 1.0)

#print(f"X shape: {X.shape}")
#print(f"Choices shape: {choices.shape}")

# Run analysis
#model = FastHB_CBC(X, choice_mask, num_samples=1000, burn_in=500, device=device)
#beta_samples, mu_samples, Sigma_samples, iter_times = model.run()

# Save results
#torch.save({
   # 'beta_samples': beta_samples.cpu(),
   # 'mu_samples': mu_samples.cpu(),
   # 'Sigma_samples': Sigma_samples.cpu(),
   # 'iter_times': iter_times
#}, 'hb_cbc_results.pt')

print("\nResults saved to hb_cbc_results.pt")