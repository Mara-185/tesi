#!/usr/bin/env python3
"""Linearity of ToT-Q relationship."""
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


FRONTENDS_PEAK = [
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
     ([23.38, 54.65, 67.1, 78.0, 81.31, 94.1]),
    ([21.76, 51.89, 66.4, 67.9, 77.55, 90.8]),
    ([22.28, 53.21, 66.13, 0, 79.45, 99.3]),
    ([24.35, 59.06, 73.3,0, 86.4, 0])]

FRONTENDS_DAC1 = [
    # OBTAINED CONSIDERING 10.1 e/DAC
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    np.array([160, 377, 480, 561, 597, 716])]

FRONTENDS_DAC2 = [
    # CONSIDERING CALIBRATION FROM IRON55
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([176, 414, 527, 616, 655, 786]),
    ([190, 448, 571, 667, 709, 850]),
    ([87, 206, 262, 307, 326, 391]),
    ([109, 256, 325, 381, 405, 485])]

FRONTENDS_DAC3 = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([175, 411, 524, 613, 651, 781]),
    ([174, 409, 521, 609, 647, 777]),
    ([83, 195, 248, 291, 309, 371]),
    ([86, 203, 259, 302, 321, 386])]


FRONTENDS_PARAM_NO = [
    # a , b, c, t , name
    (0.12, 4, 200, 20, 'Normal'),
    (0.119, 1.4, 140, 40, 'Cascode'),
    (0.257, 3.2, 160, 17, 'HV Casc.'),
    (0.275, 2.3, 140, 13, 'HV')]

FRONTENDS_PARAM_CONSTR= [
    # a , b, c, t , name
    (0.135, -0.436, 40, 48, 'Normal'),
    (0.132, -1.3, 39, 54.3, 'Cascode'),
    (0.271, 0.4, 46, 30.7, 'HV Casc.'),
    (0.287, -0.2, 36, 27.6, 'HV')]

FRONTENDS_PARAM_TH= [
    # threshold, name
    (53.62),
    (60.19),
    (35.34),
    (31.7)]

FRONT_CAL_IRON =[
    (9.2),
    (8.5),
    (18.5),
    (14.9)
]

FRONT_CAL_SOURCES =[
    (9.26),
    (9.31),
    (19.51),
    (18.75)
]





if __name__ == "__main__":

    def func_norm(x,a,b,c,t, th):
        return np.where(x<th, 0, np.maximum(0, a*x+b-(c/(x-t))))

    MARKER = [".", "+", "*", "2"]
    # COL = ["r", "b", "g", "orange"]

    dac = FRONTENDS_DAC1[0]

    # CALIBRATION 10-1 e-/DAC (EACH FE & ALL)
    #for name in ["Normal", "Cascode", "HV Casc.", "HV"]:
    #for (a,b,c,t,name), p,thr, dac, poi, color in zip(FRONTENDS_PARAM_NO, FRONTENDS_PEAK, FRONTENDS_PARAM_TH, POINT, COL):
    for (a,b,c,t,name), p,thr, point in zip(FRONTENDS_PARAM_NO, FRONTENDS_PEAK, FRONTENDS_PARAM_TH, MARKER):
        print(name)
        print(dac)
        print(p)
        print(thr)
        print(a, b, c, t)

        plt.plot(dac, np.asarray(p), f"{point}")
        y =  np.arange(thr, 1000, 10)
        plt.plot(y, func_norm(y, a, b, c, t, thr), "--")

        plt.xlim([0, 1200])
        plt.ylim([0, 128])
        plt.suptitle(f"ToT curve linearity")
        plt.title(f"{name} - C = 10.1 e-/DAC")
        plt.xlabel("True injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        plt.savefig(f"ToT_{name}_thesis.png")
        plt.clf()
        # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        # cb = integer_ticks_colorbar()
        # cb.set_label("Hits / bin")
        #plt.legend(loc="upper left")

    # ALL
    for (a,b,c,t,name), p,thr, point in zip(FRONTENDS_PARAM_NO, FRONTENDS_PEAK, FRONTENDS_PARAM_TH, MARKER):
        print(name)
        print(dac)
        print(p)
        print(thr)
        print(a, b, c, t)

        plt.plot(dac, np.asarray(p), f"{point}")
        y =  np.arange(thr, 1000, 10)
        plt.plot(y, func_norm(y, a, b, c, t, thr), "--")


    plt.xlim([0, 1200])
    plt.ylim([0, 128])
    plt.suptitle(f"ToT curve linearity")
    plt.title(f"All FE - C = 10.1 e-/DAC")
    plt.xlabel("True injected charge [DAC]")
    plt.ylabel("ToT [25 ns]")
    plt.savefig(f"Tot_linearity_thesis.png")
    plt.clf()

    del a, b, c, t, name, p, thr, dac, point


    # CALIBRATION FROM IRON PEAKS (EACH FE &ALL)
    for (a,b,c,t,name), p,thr, dac, point, calcap in zip(FRONTENDS_PARAM_NO, FRONTENDS_PEAK, FRONTENDS_PARAM_TH, FRONTENDS_DAC2, MARKER,FRONT_CAL_IRON):
        print(dac)
        print(p)
        print(thr)
        print(a, b, c, t)

        plt.plot(np.asarray(dac), np.asarray(p), f"{point}")
        y =  np.arange(thr, 1000, 1)
        plt.plot(y, func_norm(y, a, b, c, t, thr), f"--")



        plt.xlim([0, 1200])
        plt.ylim([0, 128])
        plt.suptitle(f"ToT curve linearity")
        plt.title(f"{name} - C = {calcap} e-/DAC (from Fe55)")
        plt.xlabel("True injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        plt.savefig(f"ToT_{name}_iron.png")
        plt.clf()
        # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        # cb = integer_ticks_colorbar()
        # cb.set_label("Hits / bin")
        #plt.legend(loc="upper left")

        # ALL
    for (a,b,c,t,name), p,thr, dac, point, calcap in zip(FRONTENDS_PARAM_NO, FRONTENDS_PEAK, FRONTENDS_PARAM_TH, FRONTENDS_DAC2, MARKER, FRONT_CAL_IRON):
        print(dac)
        print(p)
        print(thr)
        print(a, b, c, t)

        plt.plot(np.asarray(dac), np.asarray(p), f"{point}")
        y =  np.arange(thr, 1000, 1)
        plt.plot(y, func_norm(y, a, b, c, t, thr), f"--")

    plt.xlim([0, 1200])
    plt.ylim([0, 128])
    plt.suptitle(f"ToT curve linearity")
    plt.title(f"All FE - C = {calcap} e-/DAC (from Fe55)")
    plt.xlabel("True injected charge [DAC]")
    plt.ylabel("ToT [25 ns]")
    plt.savefig(f"Tot_linearity_iron.png")
    plt.clf()

    del a, b, c, t, name, p, thr, dac, point

    # CALIBRATION FROM SOURCES PEAKS MEANS (EACH FE &ALL)
    for (a,b,c,t,name), p,thr, dac, point, calcap2 in zip(FRONTENDS_PARAM_NO, FRONTENDS_PEAK, FRONTENDS_PARAM_TH, FRONTENDS_DAC3, MARKER, FRONT_CAL_SOURCES):
        print(dac)
        print(p)
        print(thr)
        print(a, b, c, t)

        plt.plot(np.asarray(dac), np.asarray(p), f"{point}")
        y =  np.arange(thr, 1000, 1)
        plt.plot(y, func_norm(y, a, b, c, t, thr), f"--")



        plt.xlim([0, 1200])
        plt.ylim([0, 128])
        plt.suptitle(f"ToT curve linearity")
        plt.title(f"{name} - C = {calcap2} e-/DAC (from all sources)")
        plt.xlabel("True injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        plt.savefig(f"ToT_{name}_sources.png")
        plt.clf()
        # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        # cb = integer_ticks_colorbar()
        # cb.set_label("Hits / bin")
        #plt.legend(loc="upper left")

        # ALL
    for (a,b,c,t,name), p,thr, dac, point, calcap2 in zip(FRONTENDS_PARAM_NO, FRONTENDS_PEAK, FRONTENDS_PARAM_TH, FRONTENDS_DAC3, MARKER, FRONT_CAL_SOURCES):
        print(dac)
        print(p)
        print(thr)
        print(a, b, c, t)

        plt.plot(np.asarray(dac), np.asarray(p), f"{point}")
        y =  np.arange(thr, 1000, 1)
        plt.plot(y, func_norm(y, a, b, c, t, thr), f"--")

    plt.xlim([0, 1200])
    plt.ylim([0, 128])
    plt.suptitle(f"ToT curve linearity")
    plt.title(f"All FE - C = {calcap2} e-/DAC (from all sources)")
    plt.xlabel("True injected charge [DAC]")
    plt.ylabel("ToT [25 ns]")
    plt.savefig(f"Tot_linearity__sources.png")
    plt.clf()
