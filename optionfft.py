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
import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft
from scipy.integrate import quad
from scipy.stats import gamma
from scipy.stats import norm


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
        return np.exp(1j*u*mu  - 0.5*u**2*var)
    

    def sample_path(self, T, N=200, plot=False):
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
        r, omega = self.r, self.omega
        denom = np.power((1 - 1j*theta*nu*u + 0.5*(u*sigma)**2*nu), t/nu)
        return np.exp(1j*u*(np.log(S0) + (r + omega)*t)) / denom
    

    def sample_path(self, T, N=200, terminal=True, plot=False):
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


class EuCall:
    """Creates an instance of a European Call option given an underlying
    stock process.

    All prices are calculated at time 0 and assume the underlying stock
    does not pay dividends.

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


    def monte_carlo_price(self, n=1000):
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
            raise("black_scholes_price requires underlying stock to be a GBM.")

        K, T, S = self.K, self.T, self.S
        S0, r, sigma = S.S0, S.r, S.sigma

        d2 = (np.log(S0/K) + (r - 0.5*sigma**2)*T) / sigma*np.sqrt(T)
        d1 = d2 + sigma*np.sqrt(T)
        PrITM = norm.cdf(d2)       
        delta = norm.cdf(d1)
        return S0*delta - K*np.exp(-r*T)*PrITM

   
    def cdfFTPrice(self, lower=0, upper=np.inf, weight=None, wvar=None):
        """Compute the price of the call option using the Fourier methods on
        page 2 of Carr and Madan 1999.

        Parameters
        ----------
        lower : float, optional
            The lower integration limit in calculating the probability
            of finishing in the money and delta integrals. Default value is 0.
            Specifying a small positive eps may help performance.

        upper : float, optional
            The upper integration limit in calculating the probability of
            finishing in the money and delta integrals. Default value is
            np.inf. If weight='cauchy' (see below), finite upper bound must be
            specified.

        weight : str, optional
            Specify the weight function for scipy to use in calculating the
            integral. Due to singularity at 0, weight='cauchy' may help
            performance.

        wvar : float or tuple, optional
            Variables associated with the weight used. Since the singularity
            is at 0, wvar is automatically set to 0 when using Cauchy weights.

        Returns
        -------
        C0 : float
            Call price computed using the Fourier methods to calculate the
            delta and the probability of finishing in the money (PrITM).
            The option price is then given by
                C0 = S0*Delta - Ke^{-rT}*PrITM.
        """
        S0, r, K, T = self.S.S0, self.S.r, self.K, self.T
        k = np.log(K) 
        phi = self.S.phi

        def PrITMIntegrand(u):
            """The integrand in the probability of finishing in the money.

            Parameters
            ----------
            u : float
                Value that integrand is to be evaluated at.

            Returns
            -------
            intITM : float
                Value of the integrand for the probability of finishing in the
                money evaluated at u.
            """
            return np.real(-1j*(np.exp(-1j*u*k) * phi(T, u)) / u)
        
        def deltaIntegrand(u):  
            """The integrand in the delta calculation.

            Parameters
            ----------
            u : float
                Value that integrand is to be evaluated at.

            Returns
            -------
            intDelta : float
                Value of the integrand for the delta evaluated at u.
            """
            numerator = -1j * (np.exp(-1j*u*k) * phi(T, u - 1j))
            return np.real(numerator / (u*phi(T, -1j)))

        # Compute the integrals
        if weight == "cauchy":
            wvar = 0

        intITM = quad(PrITMIntegrand, a=lower, b=upper, weight=weight, wvar=wvar)[0]
        intDelta = quad(deltaIntegrand, a=lower, b=upper, weight=weight, wvar=wvar)[0]
        PrITM = 0.5 + intITM/np.pi
        delta = 0.5 + intDelta/np.pi
        return S0*delta - K*np.exp(-r*T)*PrITM

    
    def MCallFT(self, v, alpha):
        """Compute the Fourier transform of the modified call option.

        Parameters
        ----------
        v : float
            Value at which to compute the Fourier Transform.
        
        alpha : float
            The modification parameter used in defining the modified call.
        
        Returns
        -------
        Psi : float
            The Fourier transform, Psi_{T}(v), of the modified call price
                c_{T}(k) = e^{alpha*k}*C_{T}(k),
            evaluated at v.
        """
        T, r = self.T, self.S.r
        denom = (alpha**2+alpha-v**2) + (2*alpha+1)*v*1j
        return np.exp(-r*T)*self.S.phi(T, v - (alpha+1)*1j) / denom


    def CMFTPrice(self, alpha = 1.5):
        """Computes the price of the call option using Carr and Madan's
        modified call method. Uses scipy to perform quadrature on the integral
        and provides the benchmark (the best performance) that can be obtained
        by the Fast-Fourier Transform method.

        Parameters
        ----------
        alpha : float, optional
            The modification parameter used in defining the modified call.
            Can be thought as a damping coefficient required to ensure 
            integrability of the Fourier transform of the modified call.
            Usually given as 1/4 of the upper bound on alpha.
        """
        k = np.log(self.K)
        def CMintegrand(v, alpha, k):
            return np.real(np.exp(-1j*v*k) * self.MCallFT(v, alpha))
        CMIntegral = quad(CMintegrand, 0, np.inf, args = (alpha, k))[0]
        return np.exp(-alpha*k)*CMIntegral / np.pi
    

    def VG_analytic_price(self):
        """Computes the price of the call option using the 'analytic' formula
        given in Carr, Chang and Madan 1998 (Theorem 2, pg 98)
        https://engineering.nyu.edu/sites/default/files/2018-09/CarrEuropeanFinReview1998.pdf

        Returns
        -------
        C0 : float
            The price of the call option using the user-defined function Psi,
            where Psi relies on numerical intergation. 
            
            NOTE: Although an expression for the call price without integrals
            is given in Carr, Chang and Madan 1998, it relies on the confluent
            hypergeometric function of two variables. This function has a 
            singularity at u=1 and is too difficult to implement in practice
            (Matusda 2004).
        """
        if not isinstance(self.S, VarianceGamma):
            raise("VG_analytic_price requires underlying stock to be a VG.")
        
        K, T, S = self.K, self.T, self.S
        theta, sigma, nu = S.theta, S.sigma, S.nu
        r, S0 = S.r, S.S0

        # Compute intermediate variables required for final substitution.
        # Intermediate variable formulae can be found on pg 194 of Matsuda
        # 2004: http://www.maxmatsuda.com/Papers/2004/Matsuda%20Intro%20FT%20Pricing.pdf
        zeta = -theta/sigma**2
        s = sigma/np.sqrt(1 + 0.5*theta**2*nu/sigma**2)
        alpha = zeta*s
        c1 = 0.5*nu*(alpha+s)**2
        c2 = 0.5*nu*alpha**2
        log_c = 1 + nu*(theta - 0.5*sigma**2)
        d = 1/s * (np.log(S0/K) + r*T + (T/nu)*np.log(log_c))

        def Psi(a, b, c):
            def Psi_integrand(u, a, b, c):
                return norm.cdf(a/np.sqrt(u) + b*np.sqrt(u))*gamma.pdf(u, a=c)
            return quad(Psi_integrand, a=0, b=np.inf, args=(a,b,c))[0]
        
        a1 = d*np.sqrt((1-c1)/nu)
        b1 = (alpha+s)*np.sqrt(nu/(1-c1))
        c = T/nu

        a2 = d*np.sqrt((1-c2)/nu)
        b2 = alpha*s*np.sqrt(nu/(1-c2))
        
        delta = Psi(a1, b1, c)
        PrITM = Psi(a2, b2, c)

        return S0*delta - K*np.exp(-r*T)*PrITM


def MCallFTo(S, T, v, alpha):
    """Compute the Fourier transform of the modified call option.
    Same functionality as EuCall.MCallFT, except it is defined outside
    to allow FFT functions to access it.

    Parameters
    ----------
    S : VarianceGamma or GeometricBrownianMotion
        The Stock price process of the call option.

    T : float
        Time to maturity of the call option.

    v : float
        Value at which to compute the Fourier Transform.
    
    alpha : float
        The modification parameter used in defining the modified call.
    
    Returns
    -------
    Psi : float
        The Fourier transform, Psi_{T}(v), of the modified call price
            c_{T}(k) = e^{alpha*k}*C_{T}(k),
        evaluated at v.
    """
    denom = (alpha**2+alpha-v**2) + (2*alpha+1)*v*1j
    return np.exp(-S.r*T)*S.phi(T, v - (alpha+1)*1j) / denom


def logStrikePartition(eta = 0.25, N = 4096):
    """Creates a partition of strike prices in the log-space for use in the
    FFT pricing functions.

    Parameters
    ----------
    eta : float, optional
        The spacing size used in the quadrature of the modified call's Fourier
        transform. Default value is 0.25 from Carr and Madan 1999.
    
    N : int, optional
        The number of points used in the quadrature of the modified call's
        Fourier transform. Defaul value is 2**12 = 4096 from Carr and Madan
        1999. Note, N should be a power of 2 for ideal performance in the fft.

    Returns
    -------
    b : float
        See description of k.
    lamb : float
        See description of k.
    k : array_like(float, ndim=1)
        Numpy array with N strike prices in the log space uniformly spaced in
        the interval [-b, b). lamb is the spacing size between log-strikes 
        in k.
    """
    b = np.pi/eta
    lamb = 2*np.pi/(eta*N)
    k = -b + lamb*np.arange(0, N)
    return (b, lamb, k)


def FFTPrice(S, T, L = 0, U = np.inf, alpha = 1.5, eta = 0.25, N = 4096):
    """Computes an array of call option prices using the Fast-Fourier
    transform method described in Carr and Madan 1999, for a specified range
    of strike prices.

    Parameters
    ----------
    S : VarianceGamma or GeometricBrownianMotion
        The Stock price process of the call option.

    T : float
        Time to maturity of the call option.
    
    L : float, optional
        Function returns call prices only for strikes between L and U.
        The default values are L = 0 and U = np.inf, so call prices for all
        strikes are returned unless otherwise specified.

    U : float, optional
        See above.
    
    alpha : float
        The modification parameter used in defining the modified call. See
        CMFT price for detailed description.
    
    eta : float, optional
        The spacing size used in the quadrature of the modified call's Fourier
        transform. Default value is 0.25 from Carr and Madan 1999.

    N : int, optional
        The number of points used in the quadrature of the modified call's
        Fourier transform. Defaul value is 2**12 = 4096 from Carr and Madan
        1999. Note, N should be a power of 2 for ideal performance in the fft.
    
    Returns
    -------
    callPrices : array_like(float, ndim=1)
        Call prices calculated using the FFT approach for strike prices
        between L and U.
    """
    # Create integration partition
    V = np.arange(0, N*eta, eta)

    # Create log-strike partition
    kPart = logStrikePartition(eta, N)
    b, k = kPart[0], kPart[2]

    # Compute Simpson's rule weights
    pm_one = np.empty((N,))
    pm_one[::2] = -1
    pm_one[1::2] = 1
    Weights = 3 + pm_one
    Weights[0] -= 1  # Kronecker Delta on the first weight
    Weights = (eta/3) * Weights

    #S equence to apply Fourier transform
    x = np.exp(1j*b*V) * MCallFTo(S, T, V, alpha) * Weights
    callPrices = np.real((np.exp(-alpha*k)/np.pi) * fft(x))

    #Return only the prices with strikes between L and U
    kIndices = np.logical_and(np.exp(k)>L, np.exp(k)<U)
    return callPrices[kIndices]


if __name__ == "__main__":
    # Test the analytic price against other methods.
    K = 140
    S0, r, sigma, T = 100, 0.05, 0.1, 1
    sigma, nu, theta = 0.25, 2, -0.1
    S = GeometricBrownianMotion(S0, r, sigma)
    V = VarianceGamma(S0, r, sigma, theta, nu)
    call = EuCall(K, T, V)

    print("MC\tcdfFT\tCMFT\tAnalytic")
    print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
        call.monte_carlo_price(),
        call.cdfFTPrice(),
        call.CMFTPrice(),
        call.VG_analytic_price()
    ))