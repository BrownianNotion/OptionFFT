import numpy as np
import scipy as sp
from scipy.stats import norm
from scipy.fft import fft
import sys
import matplotlib.pyplot as plt

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
        return np.exp(-0.5*self.logSig*t * u**2 + (self.logMu*t*u + np.log(self.S0)*u)*1j)
    
    #Generate a sample path of GBM with optional plot. N = number of subintervals used.
    def samplePath(self, T, N = 1000, terminal = True, plot = False):
        dt = T/N
        t = np.linspace(0, T, N + 1)
        dW = np.random.normal(0, np.sqrt(dt), N)
        W = np.insert(np.cumsum(dW), 0, 0)  #W_0 = 0 and add increments to generate Brownian Motion
        if terminal:
            return self.S0 * np.exp(self.logMu*T + self.sigma*W[-1])
        else:
            St = self.S0 * np.exp(self.logMu*t + self.sigma*W)
            if plot:
                plt.plot(t, St)
                plt.show()
            return St

#Variance-Gamma Process
class VG():
    def __init__(self, S0, r, sigma, theta, nu, omega=None):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.theta = theta
        self.nu = nu
        #Default value of omega is such that mean return is r
        if not omega:
            omega = (1/nu) * np.log(1 - theta*nu - 0.5*nu*sigma**2)
        self.omega = omega
    
    #Characteristic function of the r.v. at time t evaluated at u.
    def phi(self, t, u):
        if t < 0:
            sys.exit("Time must be positive.")
        denom = np.power((1 - 1j*self.theta*self.nu*u + 0.5*u**2 * self.sigma**2 * self.nu), T/self.nu)
        return np.exp(1j*u*(np.log(self.S0) + (r + self.omega)*t)) / denom
    
    #Generate a sample path of the VG process, same input parameters as for GBM
    def samplePath(self, T, N = 1000, terminal = True, plot = False):
        dt = T/N
        t = np.linspace(0, T, N + 1)
        Z = np.random.normal(0, 1, N)
        a, b = dt/self.nu, self.nu
        dG = np.random.gamma(a, b, N) #Gamma increments
        X = self.theta*np.cumsum(dG) + self.sigma*np.cumsum(np.sqrt(dG)*Z)
        X = np.insert(X, 0, 0) #X_0 = 0
        if terminal:
            return self.S0 * np.exp((self.r + self.omega)*T + X[-1])
        else:
            St = self.S0 * np.exp((self.r + self.omega)*t + X)
            if plot:
                plt.plot(t, X)
                plt.show()
            return St

#Vanilla European call option class
#May want to add dividends/arbitrary time later
class EuCall():
    #K - strike price, T - time to maturity, S - a stochastic process object
    #that models the underlying stock
    def __init__(self, K, T, process):
        self.K = K
        self.T = T
        self.S = process

    #Compute payoff at maturity given a terminal asset price
    def payoff(self, ST):
        return max(ST - self.K, 0)

    #Monte Carlo method - simulate n sample paths, compute the payoff, average these and discount.
    def MonteCarloPrice(self, n=1000):
        total = 0
        for i in range(n):
            ST = self.S.samplePath(T)
            total += self.payoff(ST)
        return np.exp(-self.S.r*self.T) * (total / n)

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
        intITM = sp.integrate.quad(PrITMIntegrand, 0, np.inf, args = (K, phi))[0]
        intDelta = sp.integrate.quad(deltaIntegrand, 0, np.inf, args = (K, phi))[0]
        PrITM = 0.5 + intITM/np.pi
        delta = 0.5 + intDelta/np.pi
        return self.S.S0 * delta - self.K * np.exp(-self.S.r * self.T) * PrITM

    #Fourier Transform of modified call e^(alpha * k) * C_T(k)
    def MCallFT(self, v, alpha):
        denom = alpha**2 + alpha - v**2 + (2*alpha + 1) * v *1j
        return np.exp(-self.S.r * self.T) * self.S.phi(self.T, v - (alpha + 1)*1j) / denom

    #Carr and Madan's analytic expression without using DFT to estimate integral
    def CMFTPrice(self, alpha = 1.5):
        k = np.log(self.K)
        def CMintegrand(v, alpha, k):
            return np.exp(-1j*v*k) * self.MCallFT(v, alpha)
        lambda v: np.real(CMintegrand(v, alpha, k))  #Discard imaginary part as it is insignificant
        CMIntegral = np.real(sp.integrate.quad(CMintegrand, 0, np.inf, args = (alpha, k))[0])
        return np.exp(-alpha*k) * CMIntegral/ np.pi


#Fourier Transform of modified call e^(alpha * k) * C_T(k) (version for outside call class)
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

#Initialise a GBM Process
S0, r, sigma = 100, 0.05, 0.1
S = GBM(S0, r, sigma)

#Set Call maturity
T = 1

#Test sample paths for VG and GBM
#S.samplePath(T, terminal=False, plot=True)

S0, r = 100, 0.02
sigma, nu, theta = 0.25, 2, -0.1
T = 5

#S = VG(S0, r, sigma, theta, nu)
#V.samplePath(T, terminal=False, plot=True)

#Use only a select portion of strikes between upper and lower bounds
L = 90
U = 105
#FFT Price Parameters
alpha = 1.5
eta = 2**(-2)  #default = 2**(-2) = 0.25
N = 2**12   #default = 2**12 = 4096
FFTp = FFTPrice(S, T, L, U, alpha, eta, N)
k = logStrikePartition(eta, N)[2]
#Get strikes between L and U
K = np.exp(k)
K = np.array([strike for strike in K if strike > L and strike < U])

#Compare prices
call = EuCall(0, T, S)
for (i, strike) in enumerate(K):
    call.K = strike
    print("{:.4f} {:.4f} {:.4f} {:.4f}".format(call.BlackScholesPrice(), call.cdfFTPrice(), call.MonteCarloPrice(), FFTp[i]))
    #print("{:.4f} {:.4f} {:.4f}".format(call.MonteCarloPrice(), call.cdfFTPrice(), FFTp[i]))
    