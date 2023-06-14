#!/usr/bin/env python3
"""Plots and fits related to a Am241 source peaks."""
import argparse
import glob
import os
import traceback
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
from uncertainties import ufloat
import tables as tb
from tqdm import tqdm
from plot_utils_pisa import *


def ff_Am241(x, f0, a, f1, mu1, sigma1, f2, mu2, sigma2, f3, mu3, sigma3, f4, mu4, sigma4):
    return (
        f0 * np.exp(-x / a)  # Noise/background
        + f1 * norm.pdf(x, mu1, sigma1)  # First peak
        + f2 * norm.pdf(x, mu2, sigma2)  # Second peak
        + f3 * norm.pdf(x, mu3, sigma3)  # Third peak
        + f4 * norm.pdf(x, mu4, sigma4))  # Fourth peak

def ff_Fe55(x, f0, mu0, sigma0):
    return (
    f0 * norm.pdf(x, mu0, sigma0)
    )


def main(input_file, overwrite=False):
    output_file = os.path.splitext(input_file)[0] + "_peak.pdf"
    if os.path.isfile(output_file) and not overwrite:
        return
    print("Plotting", input_file)
    with tb.open_file(input_file) as f, PdfPages(output_file) as pdf:
        cfg = get_config_dict(f)
        plt.figure(figsize=(6.4, 4.8))

        draw_summary(input_file, cfg)
        pdf.savefig(); plt.clf()
        # print("Summary")

        # Prepare histogram
        counts = np.zeros((512, 512, 128))

        # Process 100k hits at a time
        csz = 2**24
        n_hits = f.root.Dut.shape[0]
        if n_hits == 0:
            plt.annotate("No hits recorded!", (0.5, 0.5), ha='center', va='center')
            plt.gca().set_axis_off()
            pdf.savefig(); plt.clf()
            return
        for i_first in tqdm(range(0, n_hits, csz), unit="chunk"):
            i_last = min(i_first + csz, n_hits)
            hits = f.root.Dut[i_first:i_last]
            with np.errstate(all='ignore'):
                tmp, edges = np.histogramdd(
                    (hits["col"], hits["row"], (hits["te"] - hits["le"]) & 0x7f),
                    bins=[512, 512, 128], range=[[0, 512], [0, 512], [0, 128]])
                counts += tmp
                del tmp
        del hits

        # Histograms
        max_tot = np.argmax(counts, axis=2)
        for fc, lc, name in FRONTENDS:
            plt.hist(max_tot[fc:lc+1,:].reshape(-1), bins=128, range=[0, 128], label=name, rasterized=True)
            plt.xlabel("ToT of max [25 ns]")
            plt.ylabel("Pixels / bin")
            plt.title("ToT bin with the most entries (approx. peak)")
            plt.legend()
            plt.grid()
            pdf.savefig(); plt.clf()

        # Find peaks
        tot_x = edges[2][:-1]
        for col, row in [(None, None)]:  # For debugging
            if col is None and row is None:
                pixel_hits = counts.sum(axis=(0,1))
            else:
                pixel_hits = counts[col,row,:]
            total_hits = pixel_hits.sum()
            fit_cut = 12
            #################        FE55 FIT        ####################
            popt, pcov = curve_fit(
                ff_Fe55, tot_x[fit_cut:], pixel_hits[fit_cut:],
                p0= (0.3*total_hits, 20, 5))

            ############################################################
            # popt, pcov = curve_fit(
            #     ff_Am241, tot_x[fit_cut:], pixel_hits[fit_cut:],
            #     p0=(total_hits, 16,
            #         0.1*total_hits, 50, 5,
            #         0.1*total_hits, 65, 5,
            #         1e-3*total_hits, 75, 5,
            #         1e-3*total_hits, 90, 5))
            plt.step(tot_x, pixel_hits, where='mid')
            plt.plot(tot_x[fit_cut:], ff_Fe55(tot_x[fit_cut:], *popt))
            plt.title(f"Pixel (col, row) = ({'all' if col is None else col}, {'all' if row is None else row})")
            plt.suptitle("Fe55 fit")
            plt.xlabel("ToT [25 ns]")
            plt.ylabel("Hits / bin")
            pdf.savefig(); plt.clf()
            for m, s, n in zip(popt, np.sqrt(pcov.diagonal()), ["f0", "mu0", "sigma0"]):
                print(f"{n:>10s} = {ufloat(m,s)}")

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The _source_scan_interpreted.h5 file(s). If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    args = parser.parse_args()

    files = []
    if args.input_file:  # If anything was given on the command line
        for pattern in args.input_file:
            files.extend(glob.glob(pattern, recursive=True))
    else:
        files.extend(glob.glob("output_data/module_0/chip_0/*_source_scan_interpreted.h5"))
    files.sort()

    for fp in tqdm(files):
        try:
            main(fp, args.overwrite)
        except Exception:
            print(traceback.format_exc())
