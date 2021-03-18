#!/usr/bin/env python
"""Implementation of Fast Fourier Transform method for European call options.

This module provides:
    1. Classes for the Geometric Brownian Motion and Variance Gamma processes
    2. Call option pricing functions for Monte Carlo, Fourier Inversion and 
       Fast Fourier transform methods
    3. Price and time comparisons between methods

References:
-----------

Carr, P, Madan, D.B, 1999, 'Option Valuation using the Fast Fourier Transform', Journal of
Computational Finance, 2, 61-63.
http://faculty.baruch.cuny.edu/lwu/890/CarrMadan99.pdf
"""
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.stats import norm
from scipy.fft import fft


class GeometricBrownianMotion:
    """Creates an instance of the Geometric Brownian Motion, the standard
    stochastic process used to model the Stock price in the original Black-
    Scholes-Merton Model.

    Parameters / Attributes
    -----------------------
    S0 : float
        The initial stock price.

    r : float
        The continuously compounded risk-free interest rate.
    
    sigma : float
        The volatility parameter.
    """


    def __init__(self, S0, r, sigma):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
    
    
    def phi(self, t, u):
        """Evaluates the characteristic function of log(St), where St is the
        stock price given by a Geometric Brownian motion.

        Parameters
        ----------
        t : array_like(float, ndim=1)
            Time of log-stock price for charateristic function to be computed;
            usually the maturity of the call option.

        u : array_like(float, ndim=1)
            Value at which the characteristic function of log(St) is to be
            computed.

        Returns
        -------
        phi_t(u) : array_like(float, ndim=1)
            Value of characteristic function of log(St) computed at u. 
        """
        S0, r, sigma = self.S0, self.r, self.sigma
        mu = np.log(S0) + (r - 0.5*sigma**2)*t
        var = t*sigma**2
        return np.exp(-1j*u*mu  - 0.5*u**2*var)
    

    def sample_path(self, T, N = 200, plot = False):
        """Generate a sample path of Geometric Brownian motion and return
        the terminal stock price.

        Parameters
        ----------
        T : float
            Terminal time of stock process.
        
        N : int
            Number of subintervals to use when generating sample path.
        
        plot : bool, optional
            If true, plots and displays the generated sample path.

        Returns
        -------
        ST : float
            The simulated value of the terminal stock price at time T.
        """
        dt = T/N
        t = np.linspace(0, T, N + 1)
        dW = np.random.normal(0, np.sqrt(dt), N)
        W = np.insert(np.cumsum(dW), 0, 0)  

        S0, r, sigma = self.S0, self.r, self.sigma
        S_sim = S0 * np.exp((r - 0.5*sigma**2)*t + sigma*W)
        if plot:
                plt.plot(t, S_sim)
                plt.show()
        return S_sim[-1]
        

class VarianceGamma:
    """Creates an instance of a stock process, where log-returns are given by
    the Variance-Gamma process. 

    The formula for the Stock price at time t is given by:
        S_t = exp((r + ω)t + X_t(θ, σ, ν)),

    where X_t(θ, σ, ν)) is a Variance-Gamma process. 

    For further information on the Variance-Gamma process, see:
    https://engineering.nyu.edu/sites/default/files/2018-09/CarrEuropeanFinReview1998.pdf

    Parameters / Attributes
    -----------------------
    S0 : float
        The initial stock price
    
    r : float
        The continuously compounded risk-free interest rate.
    
    sigma : float
        Parameter controlling the volatility of the Variance-Gamma process.
    
    theta : float
        Parameter controlling the drift of the Variance-Gamma process.
    
    nu : float
        Parameter controlling the variance rate of Gamma process behind the
        Variance-Gamma process.
    
    omega : float, optional
        The drift correction term. Default value is
            ω = 1/ν log(1 - θν - 0.5νσ^2),
        which ensures the stock's expected return is equal to the continuously
        compounded risk free rate r.
    """
    def __init__(self, S0, r, sigma, theta, nu, omega=None):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.theta = theta
        self.nu = nu

        if omega is None:
            omega = (1/nu) * np.log(1 - theta*nu - 0.5*nu*sigma**2)
        self.omega = omega
    
   
    def phi(self, t, u):
        """Evaluates the characteristic function of log(St), where St is the
        stock price given by a Variance-Gamma process.

        Parameters
        ----------
        t : array_like(float, ndim=1)
            Time of log-stock price for charateristic function to be computed;
            usually the maturity of the call option.

        u : array_like(float, ndim=1)
            Value at which the characteristic function of log(St) is to be
            computed.

        Returns
        -------
        phi_t(u) : array_like(float, ndim=1)
            Value of characteristic function of log(St) computed at u. 
        """
        S0, theta, sigma, nu = self.S0, self.theta, self.sigma, self.nu
        omega = self.omega
        denom = np.power((1 - 1j*theta*nu*u + 0.5*(u*sigma)**2*nu), T/nu)
        return np.exp(1j*u*(np.log(S0) + (r + omega)*t)) / denom
    

    def sample_path(self, T, N = 200, terminal = True, plot = False):
        """Generate a sample path of stock based on the Variance-Gamma process
        and return the terminal stock price.

        Parameters
        ----------
        T : float
            Terminal time of stock process.
        
        N : int
            Number of subintervals to use when generating sample path.
        
        plot : bool, optional
            If true, plots and displays the generated sample path.

        Returns
        -------
        ST : float
            The simulated value of the terminal stock price at time T.
        """
        S0, theta, sigma, nu = self.S0, self.theta, self.sigma, self.nu
        r, omega = self.r, self.omega

        dt = T/N
        t = np.linspace(0, T, N + 1)
        Z = np.random.normal(0, 1, N)

        # Generate gamma increments and evaluate at Arithmetic Brownian
        # motion
        dG = np.random.gamma(dt/nu, nu, N)
        X = theta*np.cumsum(dG) + sigma*np.cumsum(np.sqrt(dG)*Z)
        X = np.insert(X, 0, 0)    # Set X_0 = 0
        S_Sim = S0 * np.exp((r + omega)*t + X)

        if plot:
            plt.plot(t, S_Sim)
            plt.show()
        return S_Sim[-1]

#Vanilla European call option class
#May want to add dividends/arbitrary time later
class EuCall:
    """Creates an instance of a European Call option given an underlying
    stock process.

    Parameters / Attributes
    -----------------------
    K : float
        Strike price of the call option.
    
    T : float
        Time to maturity of the call option.
    
    S : GeometricBrownianMotion or VarianceGamma
        An instance of the GeometricBrownianMotion or VarianceGamma classes
        to specify the process and the parameters that should be used to model
        the underlying stock.
    """
    def __init__(self, K, T, S):
        self.K = K
        self.T = T
        self.S = S


    def payoff(self, ST):
        """Computes the terminal payoff of the call option.

        Parameters
        ----------
        ST : float
            Terminal stock price.
        
        Returns
        -------
        CT : float
            Terminal payoff of the call option.
        """
        return max(ST - self.K, 0)

    #Monte Carlo method - simulate n sample paths, compute the payoff, average these and discount.
    def monte_carlo_price(self, n=1):
        """Computes the price of the call option using Monte Carlo simulation.

        Parameters
        ----------
        n : int
            Number of sample paths of the stock price to simulate.
        
        Returns
        -------
        C0 : float
            Call price approximated by averaging the discounted terminal 
            payoffs of the call option across the n simulations.
        """
        r, T = self.S.r, self.T

        total_payoff = 0
        for i in range(n):
            ST = self.S.sample_path(T)
            total_payoff += self.payoff(ST)
        return 1/n *np.exp(-r*T)*total_payoff


    def black_scholes_price(self):
        """Computes the price of the call option using the classical
        Black-Scholes formula.

        Returns
        -------
        C0 : float
            Call price computed using the Black-Scholes formula:
                C0 = S0 N(d1) - Ke^(-rT) N(d2),
            where the underlying stock process S is required to be a
            Geometric Brownian motion.
        """
        if not isinstance(self.S, GeometricBrownianMotion):
            raise("Black Scholes Pricing requires underlying stock to be a GBM.")

        K, T, S = self.K, self.T, self.S
        S0, r, sigma = S.S0, S.r, S.sigma

        d2 = (np.log(S.S0 / K) + (S.r - 0.5*sigma**2) * T) / sigma*np.sqrt(T)
        d1 = d2 + sigma*np.sqrt(T)
        PrITM = norm.cdf(d2)       
        delta = norm.cdf(d1)
        return S0*delta - K*np.exp(-r*T)*PrITM

    #bop. a
    #Fourier Transform method to compute option prices based on inversion method by Gil-Palez.
    #Expression based on rewriting the cdf using Fourier Transforms and characteristic functions
    def cdfFTPrice(self):
        #Integrands for the analytic Fourier Transform method to compute option prices
        def PrITMIntegrand(u, K, phi):
            return np.real(-1j * (np.exp(-1j*u*np.log(K)) * phi(u)) / u)
        
        def deltaIntegrand(u, K, phi):
            return np.real(-1j * (np.exp(-1j*u*np.log(K)) * phi(u - 1j)) / (u * phi(-1j)))
        
        #Estimate the required integrals
        K = self.K
        phi = lambda u: self.S.phi(self.T, u)  #Characteristic function of log-asset at maturity
        intITM = quad(PrITMIntegrand, 0, np.inf, args = (K, phi))[0]
        intDelta = quad(deltaIntegrand, 0, np.inf, args = (K, phi))[0]
        PrITM = 0.5 + intITM/np.pi
        delta = 0.5 + intDelta/np.pi
        return self.S.S0 * delta - self.K * np.exp(-self.S.r * self.T) * PrITM

    #Fourier Transform of modified call e^(alpha * k) * C_T(k)
    def MCallFT(self, v, alpha):
        denom = alpha**2 + alpha - v**2 + (2*alpha + 1) * v *1j
        return np.exp(-self.S.r * self.T) * self.S.phi(self.T, v - (alpha + 1)*1j) / denom

    #Carr and Madan's analytic expression without using DFT to estimate integral
    #Useful for testing the accuracy of the FFT approximation
    def CMFTPrice(self, alpha = 1.5):
        k = np.log(self.K)
        def CMintegrand(v, alpha, k):
            return np.real(np.exp(-1j*v*k) * self.MCallFT(v, alpha))
        CMIntegral = quad(CMintegrand, 0, np.inf, args = (alpha, k))[0]
        return np.exp(-alpha*k) * CMIntegral/ np.pi


#Fourier Transform of modified call e^(alpha * k) * C_T(k) (version outside call class for FFT)
def MCallFTo(S, T, v, alpha):
    denom = alpha**2 + alpha - v**2 + (2*alpha + 1) * v *1j
    return np.exp(-S.r * T) * S.phi(T, v - (alpha + 1)*1j) / denom

#Return a list of b (left-endpoint), lamba (log-strike spacing) and log-strike prices array k
def logStrikePartition(eta = 0.25, N = 4096):
    b = np.pi/eta
    lamb = 2*np.pi/(eta*N)
    k = -b + lamb*np.arange(0, N)
    return [b, lamb, k]

#Carr and Madan method on a lattice of log-strikes from from -pi/eta to pi/eta, right endpoint not included.
#S - stock process, T - maturity of option, L - lower bound of strike price, K - upper bound of strike price
#alpha - damping coeffcient that ensures integrability, eta - partition spacing, N - the number of partition points.
def FFTPrice(S, T, L = 0, U = np.inf, alpha = 1.5, eta = 0.25, N = 4096):
    #Create integration partition
    V = np.arange(0, N*eta, eta)

    #Create log-strike partition
    kPart = logStrikePartition(eta, N)
    b = kPart[0]
    k = kPart[2]

    #Compute Simpson's rule weights
    Weights = 3 + np.power(-1, np.arange(0, N))
    Weights[0] -= 1  #Kronecker Delta on the first weight
    Weights = (eta/3) * Weights

    #Sequence to apply Fourier transform
    x = np.exp(1j*b*V) * MCallFTo(S, T, V, alpha) * Weights
    callPrices = np.real((np.exp(-alpha*k)/np.pi) * fft(x))

    #Return only the prices with strikes between L and U
    kIndices = np.logical_and(np.exp(k)>L, np.exp(k)<U)
    return callPrices[kIndices]

########################################################################
# 3. COMPARING OPTION PRICES

#Initialise a GBM Process
S0, r, sigma = 100, 0.05, 0.1
S = GeometricBrownianMotion(S0, r, sigma)

#Set Call maturity.
T = 1

#Use only a select portion of strikes between upper and lower bounds
L = 70
U = 130
#FFT Price Parameters
alpha = 1.5
eta = 2**(-2)  #default = 2**(-2) = 0.25
N = 2**12   #default = 2**12 = 4096

#Get strikes between L and U
k = logStrikePartition(eta, N)[2]
K = np.exp(k)
K = np.array([strike for strike in K if strike > L and strike < U])

############################################################
# 3.a) TIMING PRICES USING GBM UNDERLYING
GBMmethods = ["cdfFTPrice", "black_scholes_price", "monte_carlo_price"]
GBMTimes = {method: 0 for method in GBMmethods}  #Dictionary to store average time for each method
GBMTimes["FFTPrice"] = 0

#Instance of call, strike will be constantly modified instead of creating new class
call = EuCall(0, T, S) 
runs = 10     #Number of runs to average the time over


#To be improved: use a more reliable method that tracks CPU time instead of time.time()
#Time non-FFT methods
for method in GBMmethods:
    start = time.time()
    for i in range(runs):
        for strike in K:
            call.K = strike
            getattr(EuCall, method)(call)
    end = time.time()
    GBMTimes[method] = (end - start)/runs

#Time FFT method (needs to be separate as it is not a method of EuCall)
start = time.time()
for i in range(runs):
    FFTp = FFTPrice(S, T, L, U, alpha, eta, N)
end = time.time()
GBMTimes["FFTPrice"] = (end - start)/runs

print("CPU times averaged over {} runs for options priced on a GBM underlying".format(runs))
GBMNames = ["cdfFTPrice", "BS", "MC", "FFT"]
headerSize = 8
GBMheaderRow = ''.join([method.ljust(headerSize) for method in GBMNames])  #Header row of method names
GBMtimeValues = "{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(*GBMTimes.values())
print(GBMheaderRow)
print(GBMtimeValues)

############################################################
# 3.b) TIMING PRICES USING VG UNDERLYING
VGmethods = ["cdfFTPrice", "monte_carlo_price", "CMFTPrice"]
VGTimes = {method: 0 for method in VGmethods}
VGTimes["FFTPrice"] = 0

#Change the call's process to a Variance-Gamma process as well as the maturity to 0.25.
#This is parameter combination 4 in Carr and Madan's paper
sigma, nu, theta = 0.25, 2, -0.1
V = VarianceGamma(S0, r, sigma, theta, nu)
call.S = V
call.T = 5  #For small maturities (eg. 1), the cdfFT method results in large errors.

#To be improved: use a more reliable method that tracks CPU time instead of time.time()
#Time non-FFT methods
for method in VGmethods:
    start = time.time()
    for i in range(runs):
        for strike in K:
            call.K = strike
            getattr(EuCall, method)(call)
    end = time.time()
    VGTimes[method] = (end - start)/runs

#Time FFT method (needs to be separate as it is not a method of EuCall)
start = time.time()
for i in range(runs):
    FFTp = FFTPrice(V, T, L, U, alpha, eta, N)
end = time.time()
VGTimes["FFTPrice"] = (end - start)/runs

print("CPU times averaged over {} runs for options priced on a VG underlying".format(runs))
VGNames = ["cdfFT", "MC", "CMFT", "FFT"]
VGheaderRow = ''.join([method.ljust(headerSize) for method in VGNames])  #Header row of method names
VGtimeValues = "{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(*VGTimes.values())
print(VGheaderRow)
print(VGtimeValues)
print()
print()

"""
#Small loop to compare prices. Testing purposes only.
sigma, nu, theta = 0.25, 2, -0.1
V = VarianceGamma(S0, r, sigma, theta, nu)
call = EuCall(0, T, V)
FFTp = FFTPrice(V, 5, L, U)
for (i, strike) in enumerate(K):
    call.K = strike
    #print("{:.4f} {:.4f} {:.4f} {:.4f}".format(call.black_scholes_price(), call.cdfFTPrice(), call.monte_carlo_price(), FFTp[i]))
    print("{:.4f} {:.4f} {:.4f} {:.4f}".format(call.monte_carlo_price(), call.cdfFTPrice(), call.CMFTPrice(), FFTp[i]))
"""