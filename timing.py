#!/usr/bin/env python


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
