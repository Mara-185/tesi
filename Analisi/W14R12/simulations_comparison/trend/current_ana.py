import argparse
import os
import logging
import tables as tb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from plot_utils_pisa import *
import glob
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import traceback
from tqdm import tqdm

"""We have done measurement:
    - ITHR=20, ICASN=0, 5, 15, 20;
    - ITHR=40, ICASN 0, 5, 10, 15, 20, 30;
    - ITHR=64, ICASN=0, 5, 10, 15, 20, 25, 30.

    With this script we want to plot the trends of the threshold and threshold
    dispersion in function of ICASN (fixing ITHR) and of ITHR (fixing ICASN).
    In particular we want all trends in one plot for varying ITHR
    and antoher one varying.

    The data of thresholds and threshold dispersions obtained from the fit are 
    reported by hand. """


def iplot(x,y,out_name, fmt=""):
    sub="ICASN= "+f"{x}, ITHR= {out_name[4:]}"
    plt.plot(x, y, fmt)
    plt.suptitle("Threshold vs ICASN")
    plt.title(f"{sub}")
    plt.xlabel("ICASN [DAC]")
    plt.ylabel("Threshold [DAC]")
    plt.grid()
    plt.savefig(f"{out_name}.pdf")
    plt.clf()

def disp_plot(x, y, out_name, fmt=""):
    sub="ICASN= "+f"{x}, ITHR= {out_name[4:-5]}"
    plt.plot(x, y, fmt)
    plt.suptitle("Threshold dispersion vs ICASN")
    plt.title(f"{sub}")
    plt.xlabel("ICASN [DAC]")
    plt.ylabel("Threshold dispersion [DAC]")
    plt.grid()
    plt.savefig(f"{out_name}.pdf")
    plt.clf()

def tplot(x, y, lab):
    plt.plot(x, y, label=f"{lab}")
    plt.suptitle("Threshold vs ICASN")
    plt.xlabel("ICASN [DAC]")
    plt.ylabel("Threshold [DAC]")
    plt.grid()
    plt.legend()
    plt.savefig(f"all_trends(ICASN).pdf")

def iplot2(x,y,out_name, fmt=""):
    sub="ICASN= "+f"{out_name[5:]}, ITHR= {x}"
    plt.plot(x, y, fmt)
    plt.suptitle("Threshold vs ITHR")
    plt.title(f"{sub}")
    plt.xlabel("ITHR [DAC]")
    plt.ylabel("Threshold [DAC]")
    plt.grid()
    plt.savefig(f"{out_name}.pdf")
    plt.clf()

def disp_plot2(x, y, out_name, fmt=""):
    sub="ICASN= "+f"{out_name[5:-5]}, ITHR= {x}"
    plt.plot(x, y, fmt)
    plt.suptitle("Threshold dispersion vs ITHR")
    plt.title(f"{sub}")
    plt.xlabel("ITHR [DAC]")
    plt.ylabel("Threshold dispersion [DAC]")
    plt.grid()
    plt.savefig(f"{out_name}.pdf")
    plt.clf()

def tplot2(x, y, lab):
    plt.plot(x, y, label=f"{lab}")
    plt.suptitle("Threshold vs ITHR")
    plt.xlabel("ITHR [DAC]")
    plt.ylabel("Threshold [DAC]")
    plt.grid(visible=True, axis="both")
    plt.legend()
    plt.savefig(f"all_trends(ITHR).pdf")


if __name__ == "__main__":

    #logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    #DATA:
    #ITHR = 64
    th1 = [61.43, 53.42, 50.33, 48.21, 46.70, 45.59, 46.09]
    thd1 = [2.45, 2.45, 2.45, 2.41, 2.38, 2.52, 2.50]
    ica1 = [0, 5, 10, 15, 20, 25, 30]

    #ITHR = 40
    th2 = [47.28, 41.07, 38.39, 36.65, 35.53, 33.37]
    thd2 = [2.12, 2.02, 2.03, 1.95, 1.91, 2.04]
    ica2 = [0, 5, 10, 15, 20, 30]

    #ITHR = 20
    th3 = [34.43, 28.10, 26.59, 24.66]
    thd3 = [1.95, 1.72, 1.75, 1.77]
    ica3 = ica2[:-2]

################################################################################
########################## ICASN ###############################################

    th = [th1, th2, th3]
    thd = [thd1, thd2, thd3]
    ica = [ica1, ica2, ica3]
    out = ["ithr64", "ithr40", "ithr20"]
    outdisp=[]
    for m in range(0, len(out)):
        outdisp.append(f"{out[m]}_disp")


    #Single PLOT trend THR vs ICASN
    # for x, y, outf in zip(ica, th, out):
    #     iplot(x, y, outf)
    #
    # for s, v, outd in zip(ica, thd, outdisp):
    #     disp_plot(s, v, outd, "-g")
    #
    # #All trends THR vs ICASN
    # for z, t, outfi in zip(ica, th, out):
    #      tplot(z, t, outfi)
    #
    # plt.clf()

################################################################################
############################## ITHR ############################################

    th4=[]; th5=[]; th6=[]; th7=[];
    thd4=[]; thd5=[]; thd6=[]; thd7=[];
    for i,j in zip(th, thd):
        th4.append(i[0])
        thd4.append(j[0])
        th5.append(i[1])
        thd5.append(j[1])
        th6.append(i[2])
        thd6.append(j[2])
        th7.append(i[3])
        thd7.append(j[3])

    th9 = [th4, th5, th6, th7]
    thd9 = [thd4, thd5, thd6, thd7]
    out2 = ["icasn0", "icans5", "icasn10", "icasn15"]
    outdisp2=[]
    for n in range(0, len(out2)):
        outdisp2.append(f"{out2[n]}_disp")
    ithr = [64, 40, 20]

    #Single PLOT trend THR VS ITHR
    for a, outfile in zip(th9, out2):
        iplot2(ithr, a, outfile)

    for f, outd2 in zip(thd9, outdisp2):
        disp_plot2(ithr, f, outd2, "-g")

    #All trends THR vs ICASN
    for d, e in zip(th9, out2):
        tplot2(ithr, d, e)
