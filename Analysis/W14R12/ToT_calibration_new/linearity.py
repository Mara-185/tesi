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
    ([21.76, 51.9, 63.8, 74.8, 77.55, 90.8])]

FRONTENDS_DAC0 = [
    # OBTAINED CONSIDERING 10.1 e/DAC
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    np.array([160, 377, 480, 561, 597, 716]),
    np.array([160, 377, 480, 561, 597, 716])]
#
# FRONTENDS_DAC2 = [
#     # CONSIDERING CALIBRATION FROM IRON55
#     # Fe , Am1, Am2, Am3 ,Cd, Am4, name
#     ([176, 414, 527, 616, 655, 786]),
#     ([190, 448, 571, 667, 709, 850])]

# FRONTENDS_e2 = [
#     # CONSIDERING CALIBRATION FROM IRON55
#     # Fe , Am1, Am2, Am3 ,Cd, Am4, name
#     ([1742, 4303, 5344, 6256, 6534, 7606]),
#     ([1813, 4316, 5540, 5667, 6483, 7604])]

# FRONTENDS_e3 = [
#     # CONSIDERING CALIBRATION FROM IRON55
#     # Fe , Am1, Am2, Am3 ,Cd, Am4, name
#     ([1616, 3993, 4958, 5804, 6062, 7057]),
#     ([1616, 3846, 4937, 5050, 5777, 6776])]

# FRONTENDS_DAC2 = [
#     # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
#     # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
#     # Fe , Am1, Am2, Am3 ,Cd, Am4, name
#     ([172, 406, 518, 605, 643, 772]),
#     ([180, 423, 539, 630, 670, 804])]

# FRONTENDS_DAC2 = [
#     # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
#     # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
#     # Fe , Am1, Am2, Am3 ,Cd, Am4, name
#     ([178, 391, 475, 549, 571, 658]),
#     ([179, 393, 497, 507, 576, 671])]

# FRONTENDS_DAC2 = [
#     # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
#     # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
#     # Fe , Am1, Am2, Am3 ,Cd, Am4, name
#     ([177, 390, 475, 550, 572, 660]),
#     ([178, 395, 500, 511, 580, 676])]

FRONTENDS_DAC1 = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([181, 447, 544, 636, 676, 811]),
    ([180, 423, 539, 630, 670, 804])]

FRONTENDS_DAC2 = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([179, 422, 537, 629, 668, 802]),
    ([181, 425, 542, 634, 673, 808])]

FRONTENDS_DAC3 = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([177, 417, 532, 622, 661, 793]),
    ([178, 419, 534, 625, 664, 797])]

FRONTENDS_DAC4 = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([179, 422, 537, 629, 668, 802]),
    ([180, 424, 540, 632, 671, 805])]

FRONTENDS_DAC_LAST = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([178, 421, 536, 626, 666, 799]),
    ([180, 424, 540, 632, 671, 805])]

FRONTENDS_DAC_FLAST = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([181, 426, 543, 634, 674, 809]),
    ([180, 424, 540, 632, 671, 805])]




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


FRONTENDS_TOT_FLAST = [
    # a , b, c, t , name
    (0.1228, 2.17, 132, 34.8, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]


FRONT_CAL_IRON1 =[
    (8.91),
    (9.00)]

FRONT_CAL_IRON2 =[
    (9.023),
    (8.95)]

FRONT_CAL_IRON3 =[
    (9.12),
    (9.08)]

FRONT_CAL_IRON4 =[
    (9.023),
    (8.98)]

FRONT_CAL_IRON_LAST =[
    (9.055),
    (8.98)]

FRONT_CAL_IRON_FLAST =[
    (8.938),
    (8.98)]


#####################################################################
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

FRONTENDS_DAC_FIRST = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([181, 426, 543, 634, 674, 809]),
    ([180, 423, 539, 630, 670, 804])]

FRONTENDS_DAC_SECOND = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([182, 429, 547, 640, 680, 816]),
    ([180, 423, 539, 630, 670, 804])]

FRONTENDS_DAC_THIRD_C = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([180, 423, 539, 630, 670, 804]),
    ([180, 423, 539, 630, 670, 804])]

FRONTENDS_DAC_THIRD_T = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([181, 425, 542, 634, 673, 808]),
    ([180, 423, 539, 630, 670, 804])]


######################################################

FRONTENDS_TOT_FIRST2 = [
    # a , b, c, t , name
    (0.1233, 2.01, 123, 36.3, 'Normal'),
    (0.1298, -1.02, 33.6, 59., 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

FRONTENDS_TOT_THIRD_T2 = [
    # a , b, c, t , name
    (0.1289, 0.77, 70, 44.17, 'Normal'),
    (0.1257, -0.14, 62, 52.52, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

KFIRST2 = [(8.931),
    (9.069)]

KTHIRD_T2 = [(8.973),
    (9.039)]

FRONTENDS_DAC_FIRST2 = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([181, 426, 543, 635, 675, 810]),
    ([178, 420, 535, 625, 665, 798])]

FRONTENDS_DAC_THIRD_T2 = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([180, 424, 540, 632, 672, 806]),
    ([179, 421, 536, 627, 667, 800])]

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

FRONTENDS_DAC_FOURTH_T = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([180, 424, 540, 632, 672, 806]),
    ([180, 423, 539, 630, 670, 804])]

FRONTENDS_DAC_FOURTH_C = [
    # CONSIDERING CALIBRATION FROM TOT-Q RELATIONSHIP
    # AND PEAKS SOURCES (MEANS OF ALL RESULTS)
    # Fe , Am1, Am2, Am3 ,Cd, Am4, name
    ([181, 426, 542, 634, 674, 809]),
    ([180, 423, 539, 630, 670, 804])]


##################################################################



# FRONTENDS_PARAM_NO = [
#     # a , b, c, t , name
#     (0.12, 4, 200, 20, 'Normal'),
#     (0.119, 1.4, 140, 40, 'Cascode')]
#
# FRONTENDS_TOT2 = [
#     # a , b, c, t , name
#     (0.1470, -2.72, 16, 50.2, 'Normal'),
#     (0.1402, -3.21, 17, 57.6, 'Cascode')]
#
# FRONTENDS_PARAM_CONSTR= [
#     # a , b, c, t , name
#     (0.135, -0.436, 40, 48, 'Normal'),
#     (0.132, -1.3, 39, 54.3, 'Cascode')]

FRONTENDS_PARAM_TH= [
    # threshold, name
    (53.62),
    (60.19)]



# FRONT_CAL_SOURCES =[
#     (9.26),
#     (9.31),
#     (19.51),
#     (18.75)
# ]

COL = [
    "orange",
    "green"
]

COL2 = [
    "blue",
    "red"
]
r"$\alpha$"

ANNOTATIONS1 = [
    (r"$^{55}Fe$   ", r"$^{241}Am$   ", r"$^{241}Am$   ", r"$^{241}Am$   ", r"$^{109}Cd$  ", r"$^{241}Am$   "),
    (r"$^{55}Fe$   ", r"$^{241}Am$   ", r"$^{241}Am$   ", r"   $^{241}Am$  ", r"$^{109}Cd$  ",r"$^{241}Am$   ")]

ANNOTATIONS2 = [
    (r"  $^{55}Fe$", r"   $^{241}Am$",  r"  $^{241}Am$", r"  $^{241}Am$", r"  $^{109}Cd$", r"  $^{241}Am$"),
    (r"  $^{55}Fe$", r"  $^{241}Am$", r"  $^{241}Am$", r"  $^{241}Am$   ", r"  $^{109}Cd$", r"  $^{241}Am$")]


###################################################################
################################DEVI CAMBIARE FRONT_CAL_IRONn, FRONTENDS_TOTn, FRONTENDS_DACn


if __name__ == "__main__":

    def func_norm(x,a,b,c,t, th):
        return np.where(x<th, 0, np.maximum(0, a*x+b-(c/(x-t))))

    MARKER = [".", "P", "*", "v"]
    # COL = ["r", "b", "g", "orange"]

    #dac = FRONTENDS_DAC1[0]

    # CALIBRATION 10-1 e-/DAC (EACH FE & ALL)
    #for name in ["Normal", "Cascode", "HV Casc.", "HV"]:
    #for (a,b,c,t,name), p,thr, dac, poi, color in zip(FRONTENDS_PARAM_NO, FRONTENDS_PEAK, FRONTENDS_PARAM_TH, POINT, COL):
    for (a,b,c,t,name), p,thr, dac, point, point_col, col, ann in zip(FRONTENDS_TOT_FIRST2, FRONTENDS_PEAK, FRONTENDS_PARAM_TH, FRONTENDS_DAC0, MARKER, COL2, COL, ANNOTATIONS1):
        print(name)
        #print(dac)
        print(p)
        print(thr)
        print(a, b, c, t)

        # color= "cyan",
        # color = "orange",


        plt.plot(dac, np.asarray(p), marker = f"{point}", linestyle=" ", label="All sources peaks")
        y =  np.arange(thr, 1000, 10)
        plt.plot(y, func_norm(y, a, b, c, t, thr), linestyle = "--", label=f"{name}")

        plt.xlim([0, 1200])
        plt.ylim([0, 128])
        plt.suptitle(f"ToT curve linearity")
        plt.title(f"{name} - K = 10.1 e-/DAC")
        for i, label in enumerate(ann):
            if (name=="Cascode") and (i==3):
                # "x-small"
                plt.annotate(label, (np.asarray(dac)[i], np.asarray(p)[i]), size= 9, ha="left", va="top")
            elif (name=="Cascode") and (i==4):
                plt.annotate(label, (np.asarray(dac)[i], np.asarray(p)[i]), size= 9, ha="right", va="bottom")
            elif (name=="Normal") and (i==4):
                plt.annotate(label, (np.asarray(dac)[i], np.asarray(p)[i]), size= 9, ha="right", va="bottom")
            else:
                plt.annotate(label, (np.asarray(dac)[i], np.asarray(p)[i]), size=9, ha="right", va="center")
        plt.xlabel("Signal charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        plt.legend()
        plt.savefig(f"ToT_{name}_thesis.png")
        plt.clf()
        # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        # cb = integer_ticks_colorbar()
        # cb.set_label("Hits / bin")
        #plt.legend(loc="upper left")

    # ALL
    # for (a,b,c,t,name), p,thr, point in zip(FRONTENDS_PARAM_NO, FRONTENDS_PEAK, FRONTENDS_PARAM_TH, MARKER):
    #     print(name)
    #     print(dac)
    #     print(p)
    #     print(thr)
    #     print(a, b, c, t)
    #
    #     plt.plot(dac, np.asarray(p), f"{point}", label=f"{name} sources peaks")
    #     y =  np.arange(thr, 1000, 10)
    #     plt.plot(y, func_norm(y, a, b, c, t, thr), "--", label=f"{name}")
    #
    #
    # plt.xlim([0, 1200])
    # plt.ylim([0, 128])
    # plt.suptitle(f"ToT curve linearity")
    # plt.title(f"All FE - K = 10.1 e-/DAC")
    # plt.xlabel("Charge [DAC]")
    # plt.ylabel("ToT [25 ns]")
    # plt.legend()
    # plt.savefig(f"Tot_linearity_thesis.png")
    # plt.clf()
    #
    # del a, b, c, t, name, p, thr, dac, point


    # CALIBRATION FROM IRON PEAKS (EACH FE &ALL)
    for (a,b,c,t,name), p,thr, dac, point, calcap, point_col, col, ann in zip(FRONTENDS_TOT_FIRST2, FRONTENDS_PEAK, FRONTENDS_PARAM_TH, FRONTENDS_DAC_FIRST2, MARKER,KFIRST2, COL2, COL, ANNOTATIONS2):
        print(name)
        print(dac)
        print(p)
        print(thr)
        print(a, b, c, t)

        plt.plot(np.asarray(dac), np.asarray(p),color = "green", marker = f"{point}", linestyle= " ", label="All sources peaks")
        y =  np.arange(thr, 1000, 1)
        plt.plot(y, func_norm(y, a, b, c, t, thr), color = "red" , linestyle= "--", label=f"{name}")

        for i, label in enumerate(ann):
            #print(i)
            # if name=="Normal" or name=="Cascode":
            #     plt.annotate(label, (np.asarray(dac)[i], np.asarray(p)[i]), size=9, ha="left", va="top")
            if (name=="Cascode") and (i==4):
                plt.annotate(label, (np.asarray(dac)[i], np.asarray(p)[i]), size= 9, ha="left", va="center")
            elif (name=="Cascode") and (i==3):
                plt.annotate(label, (np.asarray(dac)[i], np.asarray(p)[i]), size= 9, ha="right", va="bottom")
            elif (name=="Normal") and (i==3):
                plt.annotate(label, (np.asarray(dac)[i], np.asarray(p)[i]), size= 9, ha="left", va="top")
            elif (name=="Normal") and (i==4):
                plt.annotate(label, (np.asarray(dac)[i], np.asarray(p)[i]), size= 9, ha="left", va="center")
            else:
                plt.annotate(label, (np.asarray(dac)[i], np.asarray(p)[i]), size=9, ha="left", va="top")

        plt.xlim([0, 1200])
        plt.ylim([0, 128])
        plt.suptitle(f"ToT curve linearity")
        plt.title(f"{name} - K = {calcap} e-/DAC (from Fe55)")
        plt.xlabel("Signal charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        plt.legend()
        plt.savefig(f"ToT_{name}_iron.png")
        plt.clf()
        # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        # cb = integer_ticks_colorbar()
        # cb.set_label("Hits / bin")
        #plt.legend(loc="upper left")

        # ALL
    # for (a,b,c,t,name), p,thr, dac, point, calcap in zip(FRONTENDS_PARAM_NO, FRONTENDS_PEAK, FRONTENDS_PARAM_TH, FRONTENDS_DAC2, MARKER, FRONT_CAL_IRON):
    #     print(dac)
    #     print(p)
    #     print(thr)
    #     print(a, b, c, t)
    #
    #     plt.plot(np.asarray(dac), np.asarray(p), f"{point}", label=f"{name} sources peaks")
    #     y =  np.arange(thr, 1000, 1)
    #     plt.plot(y, func_norm(y, a, b, c, t, thr), f"--", label=f"{name}")
    #
    # plt.xlim([0, 1200])
    # plt.ylim([0, 128])
    # plt.suptitle(f"ToT curve linearity")
    # plt.title(f"All FE - K from Fe55")
    # plt.xlabel("Charge [DAC]")
    # plt.ylabel("ToT [25 ns]")
    # plt.legend()
    # plt.savefig(f"Tot_linearity_iron.png")
    # plt.clf()

    del a, b, c, t, name, p, thr, dac, point


    # CALIBRATION FROM SOURCES PEAKS MEANS (EACH FE &ALL)
    # for (a,b,c,t,name), p,thr, dac, point, calcap2 in zip(FRONTENDS_PARAM_NO, FRONTENDS_PEAK, FRONTENDS_PARAM_TH, FRONTENDS_DAC3, MARKER, FRONT_CAL_SOURCES):
    #     print(dac)
    #     print(p)
    #     print(thr)
    #     print(a, b, c, t)
    #
    #     plt.plot(np.asarray(dac), np.asarray(p), f"{point}", label="All sources peaks")
    #     y =  np.arange(thr, 1000, 1)
    #     plt.plot(y, func_norm(y, a, b, c, t, thr), f"--", label=f"{name}")
    #
    #
    #
    #     plt.xlim([0, 1200])
    #     plt.ylim([0, 128])
    #     plt.suptitle(f"ToT curve linearity")
    #     plt.title(f"{name} - C = {calcap2} e-/DAC (from all sources)")
    #     plt.xlabel("Charge [DAC]")
    #     plt.ylabel("ToT [25 ns]")
    #     plt.legend()
    #     plt.savefig(f"ToT_{name}_sources.png")
    #     plt.clf()
    #     # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
    #     # cb = integer_ticks_colorbar()
    #     # cb.set_label("Hits / bin")
    #     #plt.legend(loc="upper left")
    #
    #     # ALL
    # for (a,b,c,t,name), p,thr, dac, point, calcap2 in zip(FRONTENDS_PARAM_NO, FRONTENDS_PEAK, FRONTENDS_PARAM_TH, FRONTENDS_DAC3, MARKER, FRONT_CAL_SOURCES):
    #     print(dac)
    #     print(p)
    #     print(thr)
    #     print(a, b, c, t)
    #
    #     plt.plot(np.asarray(dac), np.asarray(p), f"{point}", label=f"{name} sources peaks")
    #     y =  np.arange(thr, 1000, 1)
    #     plt.plot(y, func_norm(y, a, b, c, t, thr), f"--", label=f"{name}")
    #
    # plt.xlim([0, 1200])
    # plt.ylim([0, 128])
    # plt.suptitle(f"ToT curve linearity")
    # plt.title(f"All FE - C from all sources")
    # plt.xlabel("Charge [DAC]")
    # plt.ylabel("ToT [25 ns]")
    # plt.legend()
    # plt.savefig(f"Tot_linearity__sources.png")
    # plt.clf()
