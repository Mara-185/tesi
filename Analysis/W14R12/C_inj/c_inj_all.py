#!/usr/bin/env python3
"""Injection capacitance calibration."""

import argparse
import glob
import os
import traceback
import itertools
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
from uncertainties import ufloat
import tables as tb
from tqdm import tqdm
#from plot_utils_pisa import *
import sys
import math

FRONTENDS_PARAM_NO = [
    # a , b, c, t , name
    (0.12, 4, 200, 20, 'Normal'),
    (0.119, 1.4, 140, 40, 'Cascode'),
    (0.257, 3.2, 160, 17, 'HV Casc.'),
    (0.275, 2.3, 140, 13, 'HV')]

FRONTENDS_PARAM_TH= [
    # threshold, name
    (53.62),
    (60.19),
    (35.34),
    (31.7)]

FRONTENDS_PEAK = [
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
     ([23.38, 54.65, 67.1, 78.0, 81.31, 94.1]),
    ([21.76, 51.89, 66.4, 67.9, 77.55, 90.8]),
    ([22.28, 53.21, 66.13, 0, 79.45, 99.3]),
    ([24.35, 59.06, 73.3,0, 86.4, 0])]

def func_norm(x,a,b,c,t, th):
    return np.where(x<th, 0, np.maximum(0, a*x+b-(c/(x-t))))

if __name__ == "__main__":
    i=1
    for (a,b,c,t,name), thr, peak in zip(FRONTENDS_PARAM_NO, FRONTENDS_PARAM_TH, FRONTENDS_PEAK):
        print(f"{name} FE:\n")
        print(peak)

        for p in peak:
            print(f"peak_tot = {p}")
            tot_dac1 = (t/2)-(b/(2*a))+(p/(2*a)) + np.sqrt(((t/2)+(b/(2*a))-(p/(2*a)))**2 + (c/a))
            tot_dac2 = (t/2)-(b/(2*a))+(p/(2*a)) - np.sqrt(((t/2)+(b/(2*a))-(p/(2*a)))**2 + (c/a))

            print(f"{i}_+ = {tot_dac1}\n")
            print(f"{i}_- = {tot_dac2}\n")
            i+=1
