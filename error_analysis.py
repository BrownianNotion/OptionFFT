#!/usr/bin/env python
"""Basic error analysis for the option pricing methods in optionfft.

Calculates average relative error over a range of prices and converts
results into LaTeX tables.
"""

import numpy as np
import pandas as pd

import optionfft as opt


# Set parameters for stock and underlying processes.
S0, r, sigma, T = 100, 0.05, 0.1, 1
sigma, nu, theta = 0.25, 2, -0.1
S = opt.GeometricBrownianMotion(S0, r, sigma)
V = opt.VarianceGamma(S0, r, sigma, theta, nu)

# Initialise instance of call, set strike to 0 since it will be
# changed in the loop.
call_GBM = opt.EuCall(0, T, S)
call_VG = opt.EuCall(0, T, V)

# FFT Price Parameters, using the defaults from Carr and Madan 1999.
alpha = 1.5
eta = 0.25
N = 4096
L, U = 60, 140      # Not interested in calls too far in or out of the money

# Get strikes between L and U
k = opt.logStrikePartition(eta, N)[2]
mask = np.logical_and(np.exp(k) > L, np.exp(k) < U)
K = np.exp(k)[mask]

# Get fft prices first.
fft_GBM = opt.FFTPrice(S, T, L, U)
fft_VG = opt.FFTPrice(V, T, L, U)

# Initialise method lists and numpy arrays to store results
GBM_methods = ["black_scholes_price", 
                "monte_carlo_price", 
                "cdfFTPrice", 
                "CMFTPrice"]
VG_methods = ["monte_carlo_price", 
              "cdfFTPrice", 
              "CMFTPrice"]
n_GBM = len(GBM_methods) + 1    
n_VG = len(VG_methods) + 1
n_prices = K.shape[0]
GBM_prices = np.zeros((n_prices, n_GBM))
VG_prices = np.zeros((n_prices, n_VG))

for (i, strike) in enumerate(K):
    call_GBM.K = strike
    call_VG.K = strike
    
    for (j, method) in enumerate(GBM_methods):
        GBM_prices[i, j] = getattr(opt.EuCall, method)(call_GBM)
    GBM_prices[i, -1] = fft_GBM[i]
    
    for (j, method) in enumerate(VG_methods):
        VG_prices[i, j] = getattr(opt.EuCall, method)(call_VG)
    VG_prices[i, -1] = fft_VG[i]

# Create pandas dataframes and export prices to csv.
GBM_names = [
    "Black-Scholes",
    "Monte Carlo",
    "Fourier Inversion",
    "Modified Call",
    "Fast-Fourier Transform"
]

VG_names = [
    "Monte Carlo",
    "Fourier Inversion",
    "Modified Call",
    "Fast-Fourier Transform"
]

# Combine prices with strikes into data frame.
all_prices = np.hstack((K.reshape(n_prices, 1), GBM_prices, VG_prices))
all_prices_names = ["Strike"] + \
                   ["GBM " + name for name in GBM_names] + \
                   ["VG " + name for name in VG_names]
all_prices_df = pd.DataFrame(data=all_prices, columns=all_prices_names)
all_prices_df.to_csv("all_prices.csv")

# Compute relative errors in the FFT price for each underlying type. 
# We use the Black-Scholes price as the theoretical for Geometric Brownian 
# motion and Fourier an average of the Fourier Inversion and Modified call 
# as the theorical for Variance Gamma process.

#GBM_df = pd.DataFrame(data=GBM_prices, columns=GBM_names)
#VG_df = pd.DataFrame(data=VG_prices, columns=VG_names)