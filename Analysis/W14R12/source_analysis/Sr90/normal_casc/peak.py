#!/usr/bin/env python3
"""Plots and fits related to a Am241 source peaks."""
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
import sys


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

def ff_Cd109(x, f0, a, f1, mu1, sigma1):
    return (
    f0 * np.exp(-x / a)  # Noise/background
    + f1 * norm.pdf(x, mu1, sigma1)
    )


def main(input_file, overwrite=False):
    output_file = os.path.splitext(input_file)[0] + "_peak2.pdf"
    if os.path.isfile(output_file) and not overwrite:
        return
    print("Plotting", input_file)


    with tb.open_file(input_file) as f, PdfPages(output_file) as pdf:
        cfg = get_config_dict(f)
        plt.figure(figsize=(6.4, 4.8))

        draw_summary(input_file, cfg)
        pdf.savefig(); plt.clf()
        # print("Summary")

        if args.npz is None:
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


            # print(counts.shape)
            # print(counts)
            #Save npz
            np.savez_compressed(
                os.path.splitext(output_file)[0] + ".npz",
                tot = counts)
            print("\"*.npz\" file is created.")

        else:
            for i,ft in enumerate(tqdm(args.npz, unit="file")):
                with np.load(ft) as data:
                    counts2d = data['counts']
                    tot1d = data['tot']
                    counts=data['counts_peak']

            edges = [np.arange(0,513,1, dtype=int),
                np.arange(0,513,1, dtype=int), np.arange(0,129,1, dtype=int)]
            #print(edges)
            print("Npz analysis...\n")
            # print(counts.shape)
            # print(counts)



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

            all_zeros = not np.any(max_tot[fc:lc+1,:])
            # print(all_zeros)

            # Enlarged plot
            if not all_zeros:
                plt.hist(max_tot[fc:lc+1,:].reshape(-1), bins=128, range=[0, 128], label=name, rasterized=True)
                plt.xlabel("ToT of max [25 ns]")
                plt.ylabel("Pixels / bin")
                plt.title("ToT bin with the most entries (approx. peak)")
                plt.xlim([0,40])
                plt.legend()
                plt.grid()
                pdf.savefig(); plt.clf()


        # Find peaks of all data
        tot_x = edges[2][:-1]
        for col, row in [(None, None)]:  # For debugging
            if col is None and row is None:
                pixel_hits = counts.sum(axis=(0,1))
            else:
                pixel_hits = counts[col,row,:]
            total_hits = pixel_hits.sum()
            fit_cut = 30
            #################        FE55 FIT        ####################
            popt, pcov = curve_fit(
                ff_Cd109, tot_x[fit_cut:], pixel_hits[fit_cut:],
                p0= (0.3*total_hits, 10, 0.1*total_hits, 75, 5))

            perr = np.sqrt(np.diag(pcov))

            ############################################################
            # popt, pcov = curve_fit(
            #     ff_Am241, tot_x[fit_cut:], pixel_hits[fit_cut:],
            #     p0=(total_hits, 16,
            #         0.1*total_hits, 50, 5,
            #         0.1*total_hits, 65, 5,
            #         1e-3*total_hits, 75, 5,
            #         1e-3*total_hits, 90, 5))
            plt.step(tot_x, pixel_hits, where='mid')
            plt.plot(tot_x[fit_cut:], ff_Cd109(tot_x[fit_cut:], *popt),
                label=f"fit {name}:\nmean={ufloat(round(popt[3], 3), round(perr[3],3))}"
                    f"\nsigma={ufloat(round(popt[4], 3),round(perr[4],3))}")
            plt.title(f"Normal & Cascode FE - Time of acquisition: 1 h + 45 minutes")
            plt.suptitle("Cd109 fit")
            plt.xlabel("ToT [25 ns]")
            plt.ylabel("Hits / bin")
            plt.legend()
            pdf.savefig(); plt.clf()
            for m, s, n in zip(popt, np.sqrt(pcov.diagonal()), ["f0", "a", "f1", "mu1", "sigma1"]):
                print(f"{n:>10s} = {ufloat(m,s)}")
            #print(total_hits)



        # Find peaks each frontend
        for fc, lc, name in FRONTENDS:
            #print(name, fc, lc)
            tot_x2 = edges[2][:-1]
            counts2 = counts[fc:lc+1,:,:]
            check = np.allclose(counts2,counts2[0])
            #print(check)
            if not check:
                for col, row in [(None, None)]:  # For debugging
                    if col is None and row is None:
                        pixel_hits2 = counts2.sum(axis=(0,1))
                    else:
                        pixel_hits = counts[col,row,:]
                    total_hits2 = pixel_hits2.sum()
                    fit_cut = 30
                    #################        FE55 FIT        ####################
                    popt, pcov = curve_fit(
                        ff_Cd109, tot_x2[fit_cut:], pixel_hits2[fit_cut:],
                        p0= (0.3*total_hits, 10, 0.1*total_hits, 75, 5))

                    perr = np.sqrt(np.diag(pcov))

                    ############################################################
                    # popt, pcov = curve_fit(
                    #     ff_Am241, tot_x[fit_cut:], pixel_hits[fit_cut:],
                    #     p0=(total_hits, 16,
                    #         0.1*total_hits, 50, 5,
                    #         0.1*total_hits, 65, 5,
                    #         1e-3*total_hits, 75, 5,
                    #         1e-3*total_hits, 90, 5))
                    plt.step(tot_x2, pixel_hits2, where='mid')
                    # plt.plot(tot_x2[fit_cut:], ff_Fe55(tot_x2[fit_cut:], *popt), "r-",
                    #     label=f"fit {name}:\nmean={ufloat(round(popt[1], 3), round(perr[1],3))}"
                    #         f"\nsigma={ufloat(round(popt[2], 3),round(perr[2],3))}")

                    plt.plot(tot_x2[fit_cut:], ff_Cd109(tot_x2[fit_cut:], *popt),
                        label=f"fit {name}:\nmean={ufloat(round(popt[3], 3), round(perr[3],3))}"
                            f"\nsigma={ufloat(round(popt[4], 3),round(perr[4],3))}")
                    plt.title("Time of acquisition: 1 h + 45 minutes")
                    #plt.title(f"Pixel (col, row) = ({'all' if col is None else col}, {'all' if row is None else row})")
                    plt.suptitle(f"Cd109 fit - {name}")
                    plt.xlabel("ToT [25 ns]")
                    plt.ylabel("Hits / bin")
                    plt.legend()
                    pdf.savefig()
                    plt.savefig(f"cd109_{name}_peak.png"); plt.clf()
                    print(f"FIT {name}:")
                    for m, s, n in zip(popt, np.sqrt(pcov.diagonal()), ["f0", "a", "f0", "mu1", "sigma1"]):
                        print(f"{n:>10s} = {ufloat(m,s)}")
                    #print(total_hits2)

        # plt.close()
        # sys.exit()



        # Analysis for each pixel
        peaks = np.full((512,512), np.nan)
        dpeaks = np.full((512,512), np.nan)
        sigma_p = np.full((512,512), np.nan)
        dsigma_p = np.full((512,512), np.nan)
        for fc, lc, name in FRONTENDS:
            print(name, fc, lc)
            print("\n")
            tot_xp = edges[2][:-1]
            countsp = counts[fc:lc+1,:,:]
            check = np.allclose(countsp,countsp[0])
            #print(check)

            if not check:
                with open("bad_pixels.txt", "w+") as pixf:
                    print("Bad pixels:", file=pixf)
                    for col, row in tqdm(itertools.product(range(224), range(512), repeat=1), desc=f"Progress in {name}: "):  # For debugging
                        #print(col,row)
                        pixel_hits = countsp[col,row,:]
                        #print(pixel_hits)
                        total_hits = pixel_hits.sum()

                        if total_hits>100:
                            print(total_hits)
                            fit_cut = 30
                            if name=="Cascode":
                                col=col+224
                            #################        FE55 FIT        ####################
                            try:
                                popt, pcov = curve_fit(
                                    ff_Fe55, tot_x2[fit_cut:], pixel_hits[fit_cut:],
                                    p0= (0.3*total_hits, 20, 5))
                            except (RuntimeError, OptimizeWarning):
                                print(f"({col,row}) - {name}", file=pixf)
                                pass
                            else:
                                perr = np.sqrt(np.diag(pcov))

                                plt.step(tot_xp, pixel_hits, where='mid')
                                plt.plot(tot_xp[fit_cut:], ff_Fe55(tot_xp[fit_cut:], *popt), "r-",
                                    label=f"fit {name}:\nmean={ufloat(round(popt[1], 3), round(perr[1],3))}"
                                        f"\nsigma={ufloat(round(popt[2], 3),round(perr[2],3))}")
                                plt.title("Time of acquisition: 2.5 hours")
                                #plt.title(f"Pixel (col, row) = ({'all' if col is None else col}, {'all' if row is None else row})")
                                plt.suptitle(f"Fe55 fit - {name} ({col,row})")
                                plt.xlabel("ToT [25 ns]")
                                plt.ylabel("Hits / bin")
                                plt.legend()
                                pdf.savefig(); plt.clf()
                                print(f"FIT {name}:")
                                for m, s, n in zip(popt, np.sqrt(pcov.diagonal()), ["f0", "mu0", "sigma0"]):
                                    print(f"{n:>10s} = {ufloat(m,s)}")
                                print(total_hits)
                                print(f"COL = {col}")

                                peaks[col,row] = popt[1]
                                dpeaks[col,row] = perr[1]
                                sigma_p[col,row] = popt[2]
                                dsigma_p[col,row] = perr[2]
                                print(peaks)
                                print(dpeaks)
                                print(sigma_p)
                                print(dsigma_p)

        np.savez_compressed(
            "tot_cd_peaks.npz",
            tot_peaks = peaks,
            dtot_peaks = dpeaks,
            tot_sigma_peaks = sigma_p,
            dtot_sigma_peaks = dsigma_p)
        print("\"*.npz\" file is created.")

        #print(peaks)
        #print(peaks[224:226,:])
                    # perr = np.sqrt(np.diag(pcov))
                    #
                    # plt.step(tot_xp, pixel_hits, where='mid')
                    # plt.plot(tot_xp[fit_cut:], ff_Fe55(tot_xp[fit_cut:], *popt), "r-",
                    #     label=f"fit {name}:\nmean={ufloat(round(popt[1], 3), round(perr[1],3))}"
                    #         f"\nsigma={ufloat(round(popt[2], 3),round(perr[2],3))}")
                    # plt.title("Time of acquisition: 2.5 hours")
                    # #plt.title(f"Pixel (col, row) = ({'all' if col is None else col}, {'all' if row is None else row})")
                    # if name=="cascode":
                    #     col=col+224
                    # plt.suptitle(f"Fe55 fit - {name} ({col,row})")
                    # plt.xlabel("ToT [25 ns]")
                    # plt.ylabel("Hits / bin")
                    # plt.legend()
                    # pdf.savefig(); plt.clf()
                    # print(f"FIT {name}:")
                    # for m, s, n in zip(popt, np.sqrt(pcov.diagonal()), ["f0", "mu0", "sigma0"]):
                    #     print(f"{n:>10s} = {ufloat(m,s)}")
                    # print(total_hits)
                    #
                    # peaks[col,row] = popt[1]
                    # dpeaks[col,row] = popt[2]
                    # sigma_p[col,row] = perr[1]
                    # dsigma_p[col,row] = perr[2]
                    # print(peaks)
                    # print(dpeaks)
                    # print(sigma_p)
                    # print(dsigma_p)


        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The _source_scan_interpreted.h5 file(s). If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    parser.add_argument("-npz", nargs="+",
                        help="the *.npz file has to be given, in order to avoid the analysis.")
    args = parser.parse_args()

    # Set OptimizeWarning
    warnings.simplefilter("error", OptimizeWarning)

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
