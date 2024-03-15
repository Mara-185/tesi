#!/usr/bin/env python3
"""Estimation of charge corresponding to ToT."""

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

# FRONTENDS_PARAM_NO = [
#     # a , b, c, t , name
#     (0.12, 4, 200, 20, 'Normal'),
#     (0.119, 1.4, 140, 40, 'Cascode'),
#     (0.257, 3.2, 160, 17, 'HV Casc.'),
#     (0.275, 2.3, 140, 13, 'HV')]
#
# FRONTENDS_TOT2 = [
#     # a , b, c, t , name
#     (0.1470, -2.72, 16, 50.2, 'Normal'),
#     (0.1402, -3.21, 17, 57.6, 'Cascode'),
#     (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
#     (0.291, -1.12, 28, 28.1, 'HV')]
#
# FRONTENDS_TOT3 = [
#     # a , b, c, t , name
#     (0.146, -2.2, 25, 48.9, 'Normal'),
#     (0.138, -2.5, 28, 56, 'Cascode'),
#     (0.2685, 0.6, 55, 29.7, 'HV Casc.'),
#     (0.2839, 0.3, 58, 25.5, 'HV')]


FRONTENDS_TOT1 = [
    # a , b, c, t , name
    (0.116, 4, 250, 22, 'Normal'),
    (0.119, 1.5, 140, 40, 'Cascode'),
    (0.247, 5.7, 330, 3, 'HV Casc.'),
    (0.269, 4.2, 260, 1, 'HV')]


FRONTENDS_TOT2 = [
    # a , b, c, t , name
    (0.138, -0.8, 60, 42.3, 'Normal'),
    (0.126, -0.2, 87, 47, 'Cascode'),
    (0.2657, 1.09, 67, 28.3, 'HV Casc.'),
    (0.2839, 0.3, 45, 27, 'HV')]


FRONTENDS_TOT3 = [
    # a , b, c, t , name
    (0.146, -2.2, 25, 48.9, 'Normal'),
    (0.138, -2.5, 28, 56, 'Cascode'),
    (0.2685, 0.6, 55, 29.7, 'HV Casc.'),
    (0.2839, 0.3, 58, 25.5, 'HV')]


FRONTENDS_TOT4 = [
    # a , b, c, t , name
    (0.1470, -2.72, 16, 50.2, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

FRONTENDS_TOT_LAST = [
    # a , b, c, t , name
    (0.145, -2.2, 24, 49.1, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

FRONTENDS_TOT_FIRST_LAST = [
    # a , b, c, t , name
    (0.1228, 2.17, 132, 34.8, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]


K1 = [(8.91),
    (9.00)]

K2 = [(9.023),
    (8.95)]

K3 = [(9.12),
    (9.08)]

K4 = [(9.023),
    (8.98)]

KLAST = [(9.055),
    (8.98)]

K_FLAST = [(8.938),
    (8.98)]

############################################################
FRONTENDS_TOT_FIRST = [
    # a , b, c, t , name
    (0.1228, 2.17, 132, 34.8, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

FRONTENDS_TOT_SECOND = [
    # a , b, c, t , name
    (0.131, 0.2, 84, 39.5, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

FRONTENDS_TOT_THIRD_C = [
    # a , b, c, t , name
    (0.142, -1.8, 29, 48.4, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

FRONTENDS_TOT_THIRD_T = [
    # a , b, c, t , name
    (0.1271, 1.11, 81, 43.17, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

KFIRST = [(8.938),
    (9.00)]

KSECOND = [(8.867),
    (8.95)]

KTHIRD_C = [(9.001),
    (9.08)]

KTHIRD_T = [(8.951),
    (8.98)]

###########################################################

FRONTENDS_TOT_FIRST2 = [
    # a , b, c, t , name
    (0.1233, 2.01, 123, 36.3, 'Normal'),
    (0.1298, -1.02, 33.6, 59., 'Cascode'),
    (0.2597, 2.63, 126, 21.1, 'HV Casc.'),
    (0.2799, 1.49, 89, 20.8, 'HV')]

FRONTENDS_TOT_THIRD_T2 = [
    # a , b, c, t , name
    (0.1289, 0.77, 70, 44.17, 'Normal'),
    (0.1257, -0.14, 62, 52.52, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

KFIRST2 = [(8.931),
    (9.069), (0), (0)]

KTHIRD_T2 = [(8.973),
    (9.039)]

################################################################

FRONTENDS_TOT_FOURTH_T = [
    # a , b, c, t , name
    (0.1395, -1.5, 20, 49.94, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

FRONTENDS_TOT_FOURTH_C = [
    # a , b, c, t , name
    (0.1435, -2.32, 19, 49.9, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

KFOURTH_T = [(8.975),
    (9.00)]

KFOURTH_C = [(8.945),
    (8.98)]


##################################################################


FRONTENDS_PARAM_TH= [
    # threshold, name
    (53.62),
    (60.19),
    (35.34),
    (31.7)]

FRONTENDS_PEAK = [
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
     ([23.38, 54.65, 67.1, 78.0, 81.31, 94.1]),
    ([21.76, 51.90, 63.84, 74.8, 77.55, 90.8]),
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



####################################
####### DEVI CAMBIARE SOLO FRONTENDS_TOTn, e Kn

if __name__ == "__main__":

    peaks = np.zeros((4, 6),dtype=float)
    tot_dac = np.zeros((4, 6),dtype=float)
    tot_dac_k = np.zeros((4, 6),dtype=float)
    tot_dac_k10 = np.zeros((4, 6),dtype=float)
    c_from_inj = np.zeros((4, 6),dtype=float)
    c_from_inj_mean = np.zeros((4, 1),dtype=float)
    new_dac = np.zeros((4, 6),dtype=float)

    with open("cap_from_injection.txt", "w+") as outf:
        for j,((a,b,c,t,name), thr, peak, k) in enumerate(zip(FRONTENDS_TOT_FIRST2, FRONTENDS_PARAM_TH, FRONTENDS_PEAK, KFIRST2)):
            print(f"{name} FE:\n")
            print(f"{name} FE:\n{peak}", file=outf)
            if name=="Normal":
                print(name, k)
            elif name=="Cascode":
                print(name ,k)
            # else:
            #     break
            print(peak)
            c_sum=0


            for i,(p,el) in enumerate(zip(peak, ELECTRONS)):
                peaks[j,i] = p

                print(f"peak_tot = {p}", file=outf)
                print(f"\npeak_tot = {p}")
                tot_dac1 = (t/2)-(b/(2*a))+(p/(2*a)) + np.sqrt(((t/2)+(b/(2*a))-(p/(2*a)))**2 + (c/a))
                #tot_dac2 = (t/2)-(b/(2*a))+(p/(2*a)) - np.sqrt(((t/2)+(b/(2*a))-(p/(2*a)))**2 + (c/a))

                print("\nCARICA OTTENUTA DALLA CURVA DI CALIBRAZIONE:")
                print(f"{i+1}_+ = {tot_dac1}\n")
                print(f"{i+1}_+ = {tot_dac1}\n", file=outf)

                q_10 = round(tot_dac1*10.1)
                print(f"ELETTRONI CON K DALLA TESI (dato {i+1}_+):")
                print(f"q{i+1}_+ = {q_10}\n")
                print(f"q{i+1}_+ = {q_10}\n", file=outf)

                print(f"percentage K_thesis: {round(((q_10-el)/el)*100,1)}\n")


                #####################################

                if name=="Normal" or name=="Cascode":
                    print(f"ELETTRONI CON K DA ANALISI (dato {i+1}_+):")
                    q_k = round(tot_dac1*k)
                    print(f"{k}, q{i+1}_+_k = {q_k}\n")
                    print(f"{k}, q{i+1}_+_k = {q_k}\n", file=outf)

                    print(f"percentage K_analysis: {round(((q_k-el)/el)*100,1)}\n")

                    q_meas = el/10.1
                    print("CARICA OTTENUTA NOTI e- E USANDO K DALLA TESI:")
                    print(f"q{i+1}_meas = {q_meas}\n")
                    print(f"q{i+1}_meas = {q_meas}\n", file=outf)

                    print("CARICA OTTENUTA NOTI e- E USANDO K DALL' ANALISI:")
                    q_meas_k = el/k
                    print(f"{k}, q{i+1}_meas_k = {q_meas_k}\n")
                    print(f"{k}, q{i+1}_meas_k = {q_meas_k}\n", file=outf)

                ########################################################



                #print(f"{i}_- = {tot_dac2}\n")
                # c_el = el/tot_dac1
                # # c_sum+=c_el
                #
                # print(f"    C = {c_el}")
                # print(f"    C = {c_el}", file=outf)



                ###############################
                if name== "Normal" or name=="Cascode":
                    if p!=0:
                        tot_dac[j, i] = tot_dac1
                        tot_dac_k10[j, i] = q_meas
                        tot_dac_k[j, i] = q_meas_k
                ###############################


                    # c_from_inj[j,i] = c_el
                    # c_sum+=c_el


                # plt.hist(c_from_inj[j])
                # plt.savefig(f"c_from_inj_ditribution_{name}.png")
                # plt.clf()


            # n = np.count_nonzero(peak)
            # c_mean=c_sum/n
            # c_from_inj_mean[j] = c_mean
            # print(n, c_mean, c_from_inj_mean)


###########################################
    if name=="Normal" or name == "Cascode":
        with open("charge_in_dac.txt", "w+") as outf2:
            print(peaks, file=outf2)
            print("DAC DA CALIBRATION CURVE:\n", file=outf2)
            print(tot_dac, file=outf2)
            print("\nDAC DA e- CON K DA TESI:\n", file=outf2)
            print(tot_dac_k10, file=outf2)
            print("\nDAC DA e- CON K DA ANALISI:\n", file=outf2)
            print(tot_dac_k, file=outf2)

            print(peaks)
            print("DAC DA CALIBRATION CURVE:\n")
            print(tot_dac)
            print("\nDAC DA e- CON K DA TESI:\n")
            print(tot_dac_k10)
            print("\nDAC DA e- CON K DA ANALISI:\n")
            print(tot_dac_k)
######################################################
    # print(c_from_inj)

    # for k, z in enumerate(c_from_inj_mean):
    #     for l,e in enumerate(ELECTRONS):
    #         new_dac[k,l] = e/z

    # print(new_dac)



    # Creating *.npz file with three field, each one of them is a matrix of four
    # rows corresponding to the FE and 6 columns (sources peaks):
    # np.savez_compressed(
    #     "cap_from_injection.npz",
    #     peaks_inj = peaks,
    #     tot_dac_inj = tot_dac,
    #     c_from_inj = c_from_inj,
    #     c_from_inj_means = c_from_inj_mean)
    # print("\"*.npz\" file is created.")
