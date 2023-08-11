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

ELECTRONS = [
    1616,
    3808,
    4849,
    5671,
    6027,
    7233,
]

def func_norm(x,a,b,c,t, th):
    return np.where(x<th, 0, np.maximum(0, a*x+b-(c/(x-t))))

if __name__ == "__main__":

    peaks = np.zeros((4, 6),dtype=float)
    tot_dac = np.zeros((4, 6),dtype=float)
    c_from_inj = np.zeros((4, 6),dtype=float)
    c_from_inj_mean = np.zeros((4, 1),dtype=float)
    new_dac = np.zeros((4, 6),dtype=float)

    with open("cap_from_injection.txt", "w+") as outf:
        for j,((a,b,c,t,name), thr, peak) in enumerate(zip(FRONTENDS_PARAM_NO, FRONTENDS_PARAM_TH, FRONTENDS_PEAK)):
            print(f"{name} FE:\n")
            print(f"{name} FE:\n{peak}", file=outf)
            print(peak)
            c_sum=0

            for i,(p,el) in enumerate(zip(peak, ELECTRONS)):
                peaks[j,i] = p
                print(f"peak_tot = {p}", file=outf)
                print(f"peak_tot = {p}")
                tot_dac1 = (t/2)-(b/(2*a))+(p/(2*a)) + np.sqrt(((t/2)+(b/(2*a))-(p/(2*a)))**2 + (c/a))
                #tot_dac2 = (t/2)-(b/(2*a))+(p/(2*a)) - np.sqrt(((t/2)+(b/(2*a))-(p/(2*a)))**2 + (c/a))

                print(f"{i+1}_+ = {tot_dac1}\n")
                print(f"{i+1}_+ = {tot_dac1}", file=outf)
                #print(f"{i}_- = {tot_dac2}\n")
                c_el = el/tot_dac1
                # c_sum+=c_el

                print(f"    C = {c_el}")
                print(f"    C = {c_el}", file=outf)
                if p!=0:
                    tot_dac[j, i] = tot_dac1
                    c_from_inj[j,i] = c_el
                    c_sum+=c_el

                plt.hist(c_from_inj[j])
                plt.savefig(f"c_from_inj_ditribution_{name}.png")
                plt.clf()


            n = np.count_nonzero(peak)
            c_mean=c_sum/n
            c_from_inj_mean[j] = c_mean
            print(n, c_mean, c_from_inj_mean)



    print(peaks)
    print(tot_dac)
    print(c_from_inj)

    for k, z in enumerate(c_from_inj_mean):
        for l,e in enumerate(ELECTRONS):
            new_dac[k,l] = e/z

    print(new_dac)



    # Creating *.npz file with three field, each one of them is a matrix of four
    # rows corresponding to the FE and 6 columns (sources peaks):
    np.savez_compressed(
        "cap_from_injection.npz",
        peaks_inj = peaks,
        tot_dac_inj = tot_dac,
        c_from_inj = c_from_inj,
        c_from_inj_means = c_from_inj_mean)
    print("\"*.npz\" file is created.")
