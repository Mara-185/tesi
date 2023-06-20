#!/usr/bin/env python3
"""Tot fit for each pixel."""
import argparse
import glob
import os
import traceback
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from scipy.stats import norm
from uncertainties import ufloat
import tables as tb
from tqdm import tqdm
from plot_utils_pisa import *

def main(infile,overwrite=False):

    #Retrieve data
    for i,fp in enumerate(tqdm(infile, unit="file")):
        with np.load(fp) as data:
            tot = data['all_tot']

    print(tot.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    parser.add_argument("-npz", nargs="+",
                        help="the *.npz file has to be given to retrieve tot data.")
    args = parser.parse_args()

    print(args.npz)

    # Set OptimizeWarning
    warnings.simplefilter("error", OptimizeWarning)



    try:
        main(args.npz, args.overwrite)
    except Exception:
        print(traceback.format_exc())
