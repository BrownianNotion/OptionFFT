import numpy as np
import scipy as sp
from scipy.stats import norm
import sys

#Geometric Brownian Motion stochastic process
class GBM():
    #S0 - initial stock price
    #r - risk-free rate
    #sigma - volatility of underlying stock
    def __init__(self, S0, r, sigma):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        #Log-Moment scaffolds. Multiply by t to get mean/var
        #of the log process
        self.logMu = r - 0.5 * sigma**2
        self.logSig = sigma**2
    
    #Characteristic function of ln(GBM) at time t, evaluated at point u
    def phi(self, t, u):
        return np.exp(-0.5 * self.logSig * t * u**2 + (self.logMu * t * u)*1j)


#Vanilla European call option class
class EuCall():
    #K - strike price, T - time to maturity, S - a stochastic process object
    #that models the underlying stock
    def __init__(self, K, T, process):
        self.K = K
        self.T = T
        self.S = process

    #Use classical Black-Scholes to price option
    #t - time 0 <= t <= T at which option is to be evaluted
    def BlackScholesPrice(self):
        if not isinstance(self.S, GBM):
            sys.exit("Black Scholes Pricing requires underlying stock to be a GBM.")

        S = self.S
        #Compute risk-neutral probability of finishing in the money
        d2 = (np.log(S.S0 / self.K) + S.logMu * self.T) / S.sigma * np.sqrt(self.T)
        PrITM = norm.cdf(d2)
        #Compute Option's delta
        d1 = d2 + S.sigma * self.T
        delta = norm.cdf(d1)
        #Price = S0 * delta - Ke^(-rT) *PrITM
        return S.S0 * delta - self.K * np.exp(-S.r * self.T) * PrITM

    #Use an inverse Fourier Transform method to get the Black-Scholes price
    def BlackScholesPriceFT(self, N):
        if not isinstance(self.S, GBM):
            sys.exit("Black Scholes Pricing requires underlying stock to be a GBM.")
        



#Quick Testing
S0, r, sigma = 1, 0.05, 0.1
S = GBM(S0, r, sigma)

t = 1
u = np.array([-1, 0, 1])
#print(S.phi(t, u))

K, T = 1, 1
call1 = EuCall(K, T, S)
#Same strike different maturities,
print(call1.BlackScholesPrice())
#print(call1.BlackScholesPrice())
