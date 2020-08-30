import math
import numpy as np
import scipy as sp

#Class object for Variance-Gamma process.
class VGProcess():
    def __init__(self, S0, r, sigma, theta, nu):
        self.S0 = S0 #initial stock price
        self.r = r #risk free rate
        self.sigma = sigma  #Jump process parameters
        self.theta = theta
        self.nu = nu
        #Default initialisation to value so that mean return is r
        omega = math.log(1 - theta * nu - 0.5 * sigma * sigma *nu) / nu
        self.omega = omega
    
    #set Omega to some other value
    def setOmega(self, omega):
        self.omega = omega

    #Characteristic function of variance Gamma process
    #u is the point of evaluation, T is time until maturity.
    def characteristicVG(u, T):
        
    