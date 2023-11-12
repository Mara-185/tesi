#!/usr/bin/env python3
"""C_injection."""
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
from plot_utils_pisa import *
import sys
import math

def main(k_file, fe, overwrite=False):

    with np.load(k_file) as k_data:
        convfac = k_data['c_inj_kalpha']
        print(convfac)

        convfac_all = convfac[0:448,:]
        print(convfac_all.shape)

        row = np.arange(512)
        col = np.arange(448)

        plt.pcolormesh(col, row, convfac_all.transpose(),
                       rasterized=True)  # Necessary for quick save and view in PDF
        #plt.title(subtitle)
        plt.suptitle("Conversion factor K map")
        plt.xlabel("Column")
        plt.ylabel("Row")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = plt.colorbar()
        cb.set_label("Conversion factor [e-/DAC]")
        frontend_names_on_top()
        plt.clim([8,11])
        plt.savefig(f"k_maps.png")
        plt.clf()


        row = np.arange(512)
        col = np.arange(224)

        # NORMAL
        convfac2 = convfac[0:224,:]
        print(convfac2.shape)


        #plt.axes((0.125, 0.11, 0.775, 0.72))
        plt.pcolormesh(col, row, convfac2.transpose(),
                       rasterized=True)  # Necessary for quick save and view in PDF
        #plt.title(subtitle)
        plt.suptitle("Conversion factor K map")
        plt.xlabel("Column")
        plt.ylabel("Row")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = plt.colorbar()
        cb.set_label("Conversion factor [e-/DAC]")
        frontend_names_on_top()
        plt.clim([8,11])
        plt.savefig(f"k_maps_Normal.png")
        plt.clf()

        #CASCODE
        convfac3 = convfac[224:448,:]
        print(convfac2.shape)

        col = np.arange(224,448)
        

        #plt.axes((0.125, 0.11, 0.775, 0.72))
        plt.pcolormesh(col, row, convfac3.transpose(),
                       rasterized=True)  # Necessary for quick save and view in PDF
        #plt.title(subtitle)
        plt.suptitle("Conversion factor K map")
        plt.xlabel("Column")
        plt.ylabel("Row")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = plt.colorbar()
        cb.set_label("Conversion factor [e-/DAC]")
        frontend_names_on_top()
        plt.clim([8,11])
        plt.savefig(f"k_maps_Cascode.png")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument( "-k", "--conversion",
        help="The npz file with the values of the conversion factor for each pixel.")
    parser.add_argument("-fe", "--frontend", help="Frontend.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    args = parser.parse_args()


    try:
        main(args.conversion, args.frontend, args.overwrite)
    except Exception:
        print(traceback.format_exc())
