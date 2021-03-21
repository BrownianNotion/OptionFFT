# OptionFFT
Implementing Fourier Transform Pricing methods for the European Call in Python,   
with a focus on the Fast Fourier transform method proposed in Carr and Madan 1999.  
Comparison of two underlying stock processes: the traditional Geometric Brownian Motion 
and the Variance-Gamma process.   
See `OptionPricing.pdf` for detailed mathematical explanations.

Classes and functions for option pricing are contained in the module `optionfft.py`.  
The script `error_analysis.py` calculates the absolute and relative errors for the FFT pricing method  
and writes the data to the tex file `err_table.tex`. The script `timing.py` computes the average time  
over a default of 10 independent runs for each method to yield the call prices in the given range of
strike prices.

## References
Carr, P, Madan, D.B, 1999, "Option Valuation using the Fast Fourier Transform", Journal of
Computational Finance, 2, 61-63.