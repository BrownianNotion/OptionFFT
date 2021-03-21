# OptionFFT
Implementing Fourier Transform Pricing methods for the European Call in Python.  
Comparison of two underlying Stock Processes: the traditional Geometric Brownian Motion and the Variance-Gamma process (Madan, Carr and Chang).  
See **OptionPricing.pdf** for detailed mathematical explanations.

Classes and functions for option pricing are contained in the module `optionfft.py`.  
The script `error_analysis.py` calculates the absolute and relative errors for the FFT pricing method  
and writes the data to the tex file `err_table.tex`.
