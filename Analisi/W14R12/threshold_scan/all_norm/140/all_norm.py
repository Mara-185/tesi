#!/usr/bin/env python3
"""Plots the results of scan_threshold (HistOcc and HistToT not required)."""
import argparse
import glob
from itertools import chain
import os
import traceback
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from tqdm import tqdm
from uncertainties import ufloat
from plot_utils_pisa import *
from scipy.optimize import curve_fit

VIRIDIS_WHITE_UNDER = matplotlib.cm.get_cmap('viridis').copy()
VIRIDIS_WHITE_UNDER.set_under('w')


@np.errstate(all='ignore')
def average(a, axis=None, weights=1, invalid=np.NaN):
    """Like np.average, but returns `invalid` instead of crashing if the sum of weights is zero."""
    return np.nan_to_num(np.sum(a * weights, axis=axis).astype(float) / np.sum(weights, axis=axis).astype(float), nan=invalid)


def main(input_file, overwrite=False):
    output_file = os.path.splitext(input_file)[0] + "_scurve.pdf"
    if os.path.isfile(output_file) and not overwrite:
        return
    print("Plotting", input_file)
    # Open file and fill histograms (actual plotting below)
    with tb.open_file(input_file) as f:
        cfg = get_config_dict(f)

        n_hits = f.root.Dut.shape[0]

        # Load information on injected charge and steps taken
        sp = f.root.configuration_in.scan.scan_params[:]
        scan_params = np.zeros(sp["scan_param_id"].max() + 1, dtype=sp.dtype)
        for i in range(len(scan_params)):
            m = sp["scan_param_id"] == i
            if np.any(m):
                scan_params[i] = sp[m.argmax()]
            else:
                scan_params[i]["scan_param_id"] = i
        del sp
        n_injections = int(cfg["configuration_in.scan.scan_config.n_injections"])
        the_vh = int(cfg["configuration_in.scan.scan_config.VCAL_HIGH"])
        start_vl = int(cfg["configuration_in.scan.scan_config.VCAL_LOW_start"])
        stop_vl = int(cfg["configuration_in.scan.scan_config.VCAL_LOW_stop"])
        step_vl = int(cfg["configuration_in.scan.scan_config.VCAL_LOW_step"])
        charge_dac_values = [
            the_vh - x for x in range(start_vl, stop_vl, step_vl)]
        subtitle = f"VH = {the_vh}, VL = {start_vl}..{stop_vl} (step {step_vl})"
        charge_dac_bins = len(charge_dac_values)
        charge_dac_range = [min(charge_dac_values) - 0.5, max(charge_dac_values) + 0.5]
        row_start = int(cfg["configuration_in.scan.scan_config.start_row"])
        row_stop = int(cfg["configuration_in.scan.scan_config.stop_row"])
        col_start = int(cfg["configuration_in.scan.scan_config.start_column"])
        col_stop = int(cfg["configuration_in.scan.scan_config.stop_column"])
        row_n, col_n = row_stop - row_start, col_stop - col_start

        # Prepare histograms
        occupancy = np.zeros((col_n, row_n, charge_dac_bins))
        tot_hist = [np.zeros((charge_dac_bins, 128)) for _ in range(len(FRONTENDS) + 1)]
        dt_tot_hist = [np.zeros((128, 479)) for _ in range(len(FRONTENDS) + 1)]
        dt_q_hist = [np.zeros((charge_dac_bins, 479)) for _ in range(len(FRONTENDS) + 1)]

        global mean
        mean =  np.zeros((4, 1), dtype="float")

        # Process one chunk of data at a time
        csz = 2**24
        for i_first in tqdm(range(0, f.root.Dut.shape[0], csz), unit="chunk"):
            i_last = min(f.root.Dut.shape[0], i_first + csz)

            # Load hits
            hits = f.root.Dut[i_first:i_last]
            with np.errstate(all='ignore'):
                tot = (hits["te"] - hits["le"]) & 0x7f
            fe_masks = [(hits["col"] >= fc) & (hits["col"] <= lc) for fc, lc, _ in FRONTENDS]

            # Determine injected charge for each hit
            vh = scan_params["vcal_high"][hits["scan_param_id"]]
            vl = scan_params["vcal_low"][hits["scan_param_id"]]
            charge_dac = vh - vl
            del vl, vh
            # Count hits per pixel per injected charge value
            occupancy_tmp, occupancy_edges = np.histogramdd(
                (hits["col"], hits["row"], charge_dac),
                bins=[col_n, row_n, charge_dac_bins],
                range=[[col_start, col_stop], [row_start, row_stop], charge_dac_range])
            occupancy_tmp /= n_injections
            occupancy += occupancy_tmp
            del occupancy_tmp

            for i, ((fc, lc, _), mask) in enumerate(zip(chain([(0, 511, 'All FEs')], FRONTENDS), chain([slice(-1)], fe_masks))):
                if fc >= col_stop or lc < col_start:
                    continue

                # ToT vs injected charge as 2D histogram
                tot_hist[i] += np.histogram2d(
                    charge_dac[mask], tot[mask], bins=[charge_dac_bins, 128],
                    range=[charge_dac_range, [-0.5, 127.5]])[0]


                # Histograms of time since previous hit vs TOT and QINJ
                # dt_tot_hist[i] += np.histogram2d(
                #     tot[mask][1:], np.diff(hits["timestamp"][mask]) / 640.,
                #     bins=[128, 479], range=[[-0.5, 127.5], [25e-3, 12]])[0]
                # dt_q_hist[i] += np.histogram2d(
                #     charge_dac[mask][1:], np.diff(hits["timestamp"][mask]) / 640.,
                #     bins=[charge_dac_bins, 479], range=[charge_dac_range, [25e-3, 12]])[0]

    # Do the actual plotting
    with PdfPages(output_file) as pdf:
        plt.figure(figsize=(6.4, 4.8))

        draw_summary(input_file, cfg)
        pdf.savefig(); plt.clf()

        if n_hits == 0:
            plt.annotate("No hits recorded!", (0.5, 0.5), ha='center', va='center')
            plt.gca().set_axis_off()
            pdf.savefig(); plt.clf()
            return

        # Look for the noisiest pixels
        top_left = np.array([[col_start, row_start]])
        max_occu = np.max(occupancy, axis=2)
        mask = max_occu > 1.05  # Allow a few extra hits
        noisy_list = np.argwhere(mask) + top_left
        noisy_indices = np.nonzero(mask)
        srt = np.argsort(-max_occu[noisy_indices])
        noisy_indices = tuple(x[srt] for x in noisy_indices)
        noisy_list = noisy_list[srt]
        if len(noisy_list):
            mi = min(len(noisy_list), 100)
            tmp = "\n".join(
                ",    ".join(f"({a}, {b}) = {float(c):.1f}" for (a, b), c in g)
                for g in groupwise(zip(noisy_list[:mi], max_occu[tuple(x[:mi] for x in noisy_indices)]), 4))
            plt.annotate(
                split_long_text(
                    "Noisiest pixels (col, row) = occupancy$_{max}$\n"
                    f"{tmp}"
                    f'{", ..." if len(noisy_list) > mi else ""}'
                    f"\nTotal = {len(noisy_list)} pixels ({len(noisy_list)/row_n/col_n:.1%})"
                ), (0.5, 0.5), ha='center', va='center')
        else:
            plt.annotate("No noisy pixel found.", (0.5, 0.5), ha='center', va='center')
        plt.gca().set_axis_off()
        pdf.savefig(); plt.clf()

        # S-Curve as 2D histogram
        occupancy_charges = occupancy_edges[2].astype(np.float32)
        occupancy_charges = (occupancy_charges[:-1] + occupancy_charges[1:]) / 2
        occupancy_charges = np.tile(occupancy_charges, (col_n, row_n, 1))
        for fc, lc, name in chain([(0, 511, 'All FEs')], FRONTENDS):
            if fc >= col_stop or lc < col_start:
                continue
            fc = max(0, fc - col_start)
            lc = min(col_n - 1, lc - col_start)
            plt.hist2d(occupancy_charges[fc:lc+1,:,:].reshape(-1),
                       occupancy[fc:lc+1,:,:].reshape(-1),
                       bins=[charge_dac_bins, 150], range=[charge_dac_range, [0, 1.5]],
                       cmin=1, rasterized=True)  # Necessary for quick save and view in PDF
            plt.title(subtitle)
            plt.suptitle(f"S-Curve ({name})")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("Occupancy")
            set_integer_ticks(plt.gca().xaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Pixels / bin")
            pdf.savefig(); plt.clf()


        # Compute the threshold for each pixel as the weighted average
        # of the injected charge, where the weights are given by the
        # occupancy such that occu = 0.5 has weight 1, occu = 0,1 have
        # weight 0, and anything in between is linearly interpolated
        # Assuming the shape is an erf, this estimator is consistent
        w = np.maximum(0, 0.5 - np.abs(occupancy - 0.5))
        threshold_DAC = average(occupancy_charges, axis=2, weights=w, invalid=0)

        # Threshold hist
        m1 = int(max(charge_dac_range[0], threshold_DAC.min() - 2))
        m2 = int(min(charge_dac_range[1], threshold_DAC.max() + 2))
        for i, (fc, lc, name) in enumerate(FRONTENDS):
            if fc >= col_stop or lc < col_start:
                continue
            fc = max(0, fc - col_start)
            lc = min(col_n - 1, lc - col_start)
            th = threshold_DAC[fc:lc+1,:]
            th_mean = ufloat(np.mean(th[th>0]), np.std(th[th>0], ddof=1))
            #bin_height, bin_edge, _ = plt.hist(th.reshape(-1), bins=m2-m1, range=[m1, m2],
            #         label=f"{name} ${th_mean:L}$", histtype='step', color=f"C{i}")
            bin_height, bin_edge, _ = plt.hist(th.reshape(-1), bins=m2-m1, range=[m1, m2],
                     label=f"{name}", histtype='step', color=f"C{i}")
##############################################

            low = [10000,10, 0]
            up = [100500, 200, 30]
            bin_center = (bin_edge[:-1]+bin_edge[1:])/2
            popt, pcov = curve_fit(gauss, bin_center, bin_height, p0 = [16000, 60, 3], bounds=(low, up))
            print(*popt)
            mean[i] = popt[1]
            perr = np.sqrt(np.diag(pcov))
            print(*perr)
            xb = np.arange(bin_edge[0], bin_edge[-1], 0.005)
            plt.plot(xb, gauss(xb, *popt), "r-", label=f"fit {name}:\nmean={ufloat(round(popt[1], 3), round(perr[1],3))}\nsigma={ufloat(round(popt[2], 3), round(perr[2],3))}")
            #Save results in a txt file
            with open(f"th_fitresults_{the_vh}[all norm].txt", "w") as outf:
                print("#A#mean#sigma:", file=outf)
                print(*popt, file=outf)
                print("#SA#Smean#Ssigma:", file=outf)
                print(*perr, file=outf)

###################################################
        plt.title(subtitle)
        plt.suptitle("Threshold distribution")
        plt.xlabel("Threshold [DAC]")
        plt.ylabel("Pixels / bin")
        set_integer_ticks(plt.gca().yaxis)
        plt.legend(loc="upper left", fontsize=9)
        plt.grid(axis='y')
        pdf.savefig(); plt.clf()


        # ToT vs injected charge as 2D histogram
        for i, ((fc, lc, name), hist) in enumerate(zip(chain([(0, 511, 'All FEs')], FRONTENDS), tot_hist)):
            if fc >= col_stop or lc < col_start:
                continue
            plt.pcolormesh(
                occupancy_edges[2], np.linspace(-0.5, 127.5, 129, endpoint=True),
                hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF
            plt.title(subtitle)
            plt.suptitle(f"ToT curve ({name})")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            pdf.savefig(); plt.clf()

            # ToT mean vs injected charge
            tot_temp = np.tile(np.linspace(0, 127, 128, endpoint=True), (charge_dac_bins,1))
            tot_mean= np.sum(tot_temp*hist,axis=1)/ np.sum(hist, axis=1)
            del tot_temp

            plt.plot(occupancy_edges[2][:-1], tot_mean, rasterized=True)
            plt.title(subtitle)
            plt.suptitle(f"ToT curve ({name})")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            pdf.savefig(); plt.clf()

            #FIT
            if name!="All FEs":
                print(i, mean[i-1])
                def func_casc(x,a,b,c,t):
                    return np.where(x<mean[i-1], 0, np.maximum(0, a*x+b-(c/(x-t))))
                low = [0 , -10, 0, 0]
                up = [0.2, 10,1000, 80]

                mask_tot = np.isfinite(tot_mean)
                tot_mean = tot_mean[mask_tot]
                occu = occupancy_edges[2][:-1]
                charge_dac_bins = occu[mask_tot]

                popt, pcov = curve_fit(func_casc, charge_dac_bins, tot_mean,
                    p0 = [0.15, 2, 10, 10],bounds=(low, up), maxfev=10000)
                perr = np.sqrt(np.diag(pcov))

                print(*popt)
                print(*perr)

                plt.pcolormesh(
                    occupancy_edges[2], np.linspace(-0.5, 127.5, 129, endpoint=True),
                    hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

                y = np.arange(mean[i-1]-0.01, 250, 1)
                plt.plot(y, func_casc(y, *popt), "r-",
                    label=f"fit {name}:\na ={ufloat(round(popt[0], 3), round(perr[0], 3))}\nb = {ufloat(round(popt[1],3),round(perr[1],3))}\nc = {ufloat(round(popt[2],3),round(perr[2], 3))}\nt = {ufloat(np.around(popt[3],3),round(perr[3], 3))}")
                plt.xlim([0, 250])
                plt.ylim([0, 40])

                plt.title(subtitle)
                plt.suptitle("ToT curve fit")
                plt.xlabel("True injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.legend(loc="upper right", fontsize=8)
                pdf.savefig(); plt.clf()


        #Threshold map

        # plt.axes((0.125, 0.11, 0.775, 0.72))
        # plt.pcolormesh(occupancy_edges[0], occupancy_edges[1], threshold_DAC.transpose(),
        #                rasterized=True)  # Necessary for quick save and view in PDF
        # plt.title(subtitle)
        # plt.suptitle("Threshold map")
        # plt.xlabel("Column")
        # plt.ylabel("Row")
        # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        # cb = plt.colorbar()
        # cb.set_label("Threshold [DAC]")
        # frontend_names_on_top()
        # pdf.savefig(); plt.clf()

        #Compute the noise (the width of the up-slope of the s-curve)
        #as a variance with the weights above

        # noise_DAC = np.sqrt(average((occupancy_charges - np.expand_dims(threshold_DAC, -1))**2, axis=2, weights=w, invalid=0))
        # del w
        #
        # # Noise hist
        # m = int(np.ceil(noise_DAC.max(initial=0, where=np.isfinite(noise_DAC)))) + 1
        # for i, (fc, lc, name) in enumerate(FRONTENDS):
        #     if fc >= col_stop or lc < col_start:
        #         continue
        #     fc = max(0, fc - col_start)
        #     lc = min(col_n - 1, lc - col_start)
        #     ns = noise_DAC[fc:lc+1,:]
        #     noise_mean = ufloat(np.mean(ns[ns>0]), np.std(ns[ns>0], ddof=1))
        #     plt.hist(ns.reshape(-1), bins=min(20*m, 100), range=[0, m],
        #              label=f"{name} ${noise_mean:L}$", histtype='step', color=f"C{i}")
        # plt.title(subtitle)
        # plt.suptitle(f"Noise (width of s-curve slope) distribution")
        # plt.xlabel("Noise [DAC]")
        # plt.ylabel("Pixels / bin")
        # set_integer_ticks(plt.gca().yaxis)
        # plt.grid(axis='y')
        # plt.legend()
        # pdf.savefig(); plt.clf()

        #Noise map

        # plt.axes((0.125, 0.11, 0.775, 0.72))
        # plt.pcolormesh(occupancy_edges[0], occupancy_edges[1], noise_DAC.transpose(),
        #                rasterized=True)  # Necessary for quick save and view in PDF
        # plt.title(subtitle)
        # plt.suptitle("Noise (width of s-curve slope) map")
        # plt.xlabel("Column")
        # plt.ylabel("Row")
        # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        # cb = plt.colorbar()
        # cb.set_label("Noise [DAC]")
        # frontend_names_on_top()
        # pdf.savefig(); plt.clf()

        # Time since previous hit vs ToT

        # for (fc, lc, name), hist in zip(chain([(0, 511, 'All FEs')], FRONTENDS), dt_tot_hist):
        #     if fc >= col_stop or lc < col_start:
        #         continue
        #     plt.pcolormesh(
        #         np.linspace(-0.5, 127.5, 129, endpoint=True),
        #         np.linspace(25e-3, 12, 480, endpoint=True),
        #         hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)
        #     plt.title(subtitle)
        #     plt.suptitle(f"Time between hits ({name})")
        #     plt.xlabel("ToT [25 ns]")
        #     plt.ylabel("$\\Delta t_{{token}}$ from previous hit [μs]")
        #     set_integer_ticks(plt.gca().xaxis)
        #     cb = integer_ticks_colorbar()
        #     cb.set_label("Hits / bin")
        #     pdf.savefig(); plt.clf()

        # Time since previous hit vs injected charge

        # m = 32 if tot.max() <= 32 else 128
        # for (fc, lc, name), hist in zip(chain([(0, 511, 'All FEs')], FRONTENDS), dt_q_hist):
        #     if fc >= col_stop or lc < col_start:
        #         continue
        #     plt.pcolormesh(
        #         occupancy_edges[2],
        #         np.linspace(25e-3, 12, 480, endpoint=True),
        #         hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)
        #     plt.title(subtitle)
        #     plt.suptitle(f"Time between hits ({name})")
        #     plt.xlabel("Injected charge [DAC]")
        #     plt.ylabel("$\\Delta t_{{token}}$ from previous hit [μs]")
        #     set_integer_ticks(plt.gca().xaxis)
        #     cb = integer_ticks_colorbar()
        #     cb.set_label("Hits / bin")
        #     pdf.savefig(); plt.clf()

        plt.close()

def gauss(x, A, mean, sigma):
    return  A * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The _threshold_scan_interpreted.h5 file(s)."
             " If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    args = parser.parse_args()

    files = []
    if args.input_file:  # If anything was given on the command line
        for pattern in args.input_file:
            files.extend(glob.glob(pattern, recursive=True))
    else:
        files.extend(glob.glob("output_data/module_0/chip_0/*_threshold_scan_interpreted.h5"))
    files.sort()

    for fp in tqdm(files, unit="file"):
        try:
            main(fp, args.overwrite)
        except Exception:
            print(traceback.format_exc())
