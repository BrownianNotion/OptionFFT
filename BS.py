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
#May want to add dividends later
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

    #Fourier Transform method to compute option prices (Scott 1997)
    def BlackScholesPriceFT(self):
        if not isinstance(self.S, GBM):
            sys.exit("Black Scholes Pricing requires underlying stock to be a GBM.")

        #Integrands for the analytic Fourier Transform method to compute option prices
        def PrITMIntegrand(u, K, phi):
            return np.real(-1j * (np.exp(-1j*u*np.log(K)) * phi(u)) / u)
        
        def deltaIntegrand(u, K, phi):
            return np.real(-1j * (np.exp(-1j*u*np.log(K)) * phi(u - 1j)) / (u * phi(-1j)))
        
        #Estimate the required integrals
        K = self.K
        phi = lambda u: self.S.phi(self.T, u)  #Characteristic function of log-asset at maturity
        intITM = sp.integrate.quad(PrITMIntegrand, 0, np.inf, args = (K, phi))[0]
        intDelta = sp.integrate.quad(deltaIntegrand, 0, np.inf, args = (K, phi))[0]

        PrITM = 0.5 + intITM / np.pi
        delta = 0.5 + intDelta / np.pi
        return self.S.S0 * delta - self.K * np.exp(-self.S.r * self.T) * PrITM




#Quick Testing
S0, r, sigma = 1, 0.05, 0.1
S = GBM(S0, r, sigma)

u = np.array([-1, 0, 1])
#print(S.phi(t, u))

K, T = 1, 1
call1 = EuCall(K, T, S)
#Same strike different maturities,
print("BS Price " + str(call1.BlackScholesPrice()))
print("FT Price " + str(call1.BlackScholesPriceFT()))

Ks = np.arange(0.5, 1, 0.1)
T = np.arange(1, 4)


for k in Ks:
    for t in T:
        Eucall1 = EuCall(k, t, S)
        err = abs(Eucall1.BlackScholesPrice() - Eucall1.BlackScholesPriceFT())
        print("err = " + str(err))
    print()