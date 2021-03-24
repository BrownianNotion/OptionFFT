#!/usr/bin/env python3
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

# Not interested in calls too far in or out of the money
lower_factor = 0.5
upper_factor = 2
L, U = lower_factor*S0, upper_factor*S0       

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
VG_methods = ["VG_analytic_price",
              "monte_carlo_price", 
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
    "Analytic",
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
all_prices_df.to_csv("Analysis/all_prices.csv")

GBM_df = pd.DataFrame(data=GBM_prices, columns=GBM_names)
VG_df = pd.DataFrame(data=VG_prices, columns=VG_names)

# Compute average absolute and relative errors in the FFT price for each 
# underlying type.
abs_GBM = abs(GBM_df["Black-Scholes"] - GBM_df["Fast-Fourier Transform"])
rel_GBM = abs_GBM/GBM_df["Black-Scholes"]

# Theoretical price under Variance Gamma is made difficult by inaccuracy
# of VG_analytic_price due to singularity at u=0, especially for ATM/OTM
# calls (Matsuda 2004)
# Hence, if Monte carlo and VG prices differ by more than threshold, use
# an average of Fourier inversion and modified call
diff_threshold = 0.03
percent_diff_mc_analytic = np.abs((VG_df["Monte Carlo"]-VG_df["Analytic"])/\
                                  VG_df["Monte Carlo"])
avg_non_analytic_prices = (VG_df["Fourier Inversion"] + VG_df["Modified Call"])/2
VG_theoretical = np.where(
                    percent_diff_mc_analytic > diff_threshold,
                    avg_non_analytic_prices,
                    VG_df["Analytic"]
                    )
VG_df["Theoretical"] = VG_theoretical
abs_VG  = abs(VG_theoretical - VG_df["Fast-Fourier Transform"])
rel_VG = abs_VG/VG_theoretical

# Convert to LaTeX table
err_data = np.array([
    [abs_GBM.mean(), rel_GBM.mean()],
    [abs_VG.mean(), rel_VG.mean()]
])

err_df = pd.DataFrame(err_data, 
                      index=["GBM", "VG"], 
                      columns=["Absolute", "Relative"])

# Change formatting of numbers in table
err_df["Absolute"] = err_df["Absolute"].apply("{:.4e}".format)
err_df["Relative"] = err_df["Relative"].apply("{:.2%}".format)

# Create tex
caption = "The absolute and relative errors of the FFT pricing method\
           computed over {:d} strike prices between {:.2f} and {:.2f}."\
            .format(n_prices, K[0], K[-1])

table_tex = err_df.to_latex(caption=caption, label="tab:err_table")

# Replace rules with hline for consistency and bold headings
table_tex = table_tex.replace(r"\toprule", r"\hline" + "\n" + r"\hline")
table_tex = table_tex.replace(r"\midrule", r"\hline")
table_tex = table_tex.replace(r"\bottomrule", r"\hline")
table_tex = table_tex.replace("Relative", r"\textbf{Relative}")
table_tex = table_tex.replace("Absolute", r"\textbf{Absolute}")
table_tex = table_tex.replace(r"\begin{table}", r"\begin{table}[h]")

# Write to table.tex
f = open("Analysis/error_table.tex", "w+")
f.write(table_tex)
f.close()