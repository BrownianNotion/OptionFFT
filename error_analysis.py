#!/usr/bin/env python
"""Basic error analysis for the option pricing methods in optionfft.

Calculates average relative error over a range of prices and converts
results into LaTeX tables.
"""

import numpy as np
import optionfft as opt


# Compare option prices of each method if run as script.
# Set parameters for stock and underlying processes.
S0, r, sigma, T = 100, 0.05, 0.1, 1
sigma, nu, theta = 0.25, 2, -0.1
S = opt.GeometricBrownianMotion(S0, r, sigma)
V = opt.VarianceGamma(S0, r, sigma, theta, nu)

# Initialise instance of call, set strike to 0 since it will be
# changed in the loop.
#call = EuCall(0, T, S)
call = opt.EuCall(0, T, V)

# FFT Price Parameters, using the defaults from Carr and Madan 1999.
alpha = 1.5
eta = 0.25
N = 4096
L, U = 80, 110
# Get strikes between L and U
k = opt.logStrikePartition(eta, N)[2]
K = np.exp(k)
K = np.array([strike for strike in K if strike > L and strike < U])
FFTp = opt.FFTPrice(V, T, L, U)

print("MC\tCDFT\tCMFT, FFT")
print_GBM = False
for (i, strike) in enumerate(K):
    call.K = strike
    if print_GBM:
        prices_str = "{:.4f} {:.4f} {:.4f} {:.4f}".format(
                        call.black_scholes_price(), 
                        call.cdfFTPrice(), 
                        call.monte_carlo_price(), 
                        FFTp[i]
                        )
    else:
        prices_str = "{:.4f} {:.4f} {:.4f} {:.4f}".format(
                        call.monte_carlo_price(), 
                        call.cdfFTPrice(),
                        call.CMFTPrice(),
                        FFTp[i]
                        )

    print(prices_str)