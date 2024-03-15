#!/usr/bin/env python3
"""Plots the results of multiple scan_threshold from the .npz files produced by plot_scurve_pisa.py."""
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
from scipy.special import erf
import sys

import warnings

VIRIDIS_WHITE_UNDER = matplotlib.cm.get_cmap('viridis').copy()
VIRIDIS_WHITE_UNDER.set_under('w')

#@np.errstate(all='ignore')
# def my_mean(a):
#     # I expect to see RuntimeWarnings in this block
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=RuntimeWarning)
#         return np.mean(a)


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

        # Process one chunk of data at a time
        csz = 2**24
        for i_first in tqdm(range(0, n_hits, csz), unit="chunk", disable=n_hits/csz<=1):
            i_last = min(n_hits, i_first + csz)

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
                dt_tot_hist[i] += np.histogram2d(
                    tot[mask][1:], np.diff(hits["timestamp"][mask]) / 640.,
                    bins=[128, 479], range=[[-0.5, 127.5], [25e-3, 12]])[0]
                dt_q_hist[i] += np.histogram2d(
                    charge_dac[mask][1:], np.diff(hits["timestamp"][mask]) / 640.,
                    bins=[charge_dac_bins, 479], range=[charge_dac_range, [25e-3, 12]])[0]

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

        # ToT vs injected charge as 2D histogram
        for (fc, lc, name), hist in zip(chain([(0, 511, 'All FEs')], FRONTENDS), tot_hist):
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

        # Compute the threshold for each pixel as the weighted average
        # of the injected charge, where the weights are given by the
        # occupancy such that occu = 0.5 has weight 1, occu = 0,1 have
        # weight 0, and anything in between is linearly interpolated
        # Assuming the shape is an erf, this estimator is consistent
        w = np.maximum(0, 0.5 - np.abs(occupancy - 0.5))
        threshold_DAC = average(occupancy_charges, axis=2, weights=w, invalid=0)


        # Time since previous hit vs ToT
        for (fc, lc, name), hist in zip(chain([(0, 511, 'All FEs')], FRONTENDS), dt_tot_hist):
            if fc >= col_stop or lc < col_start:
                continue
            plt.pcolormesh(
                np.linspace(-0.5, 127.5, 129, endpoint=True),
                np.linspace(25e-3, 12, 480, endpoint=True),
                hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)
            plt.title(subtitle)
            plt.suptitle(f"Time between hits ({name})")
            plt.xlabel("ToT [25 ns]")
            plt.ylabel("$\\Delta t_{{token}}$ from previous hit [μs]")
            set_integer_ticks(plt.gca().xaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            pdf.savefig(); plt.clf()

        # Time since previous hit vs injected charge
        m = 32 if tot.max() <= 32 else 128
        for (fc, lc, name), hist in zip(chain([(0, 511, 'All FEs')], FRONTENDS), dt_q_hist):
            if fc >= col_stop or lc < col_start:
                continue
            plt.pcolormesh(
                occupancy_edges[2],
                np.linspace(25e-3, 12, 480, endpoint=True),
                hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)
            plt.title(subtitle)
            plt.suptitle(f"Time between hits ({name})")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("$\\Delta t_{{token}}$ from previous hit [μs]")
            set_integer_ticks(plt.gca().xaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            pdf.savefig(); plt.clf()

        plt.close()

        # Save some data
        threshold_data = np.full((512, 512), np.nan)
        threshold_data[col_start:col_stop,row_start:row_stop] = threshold_DAC
        noise_data = np.full((512, 512), np.nan)
        noise_data[col_start:col_stop,row_start:row_stop] = noise_DAC
        np.savez_compressed(
            os.path.splitext(output_file)[0] + ".pdf",
            thresholds=threshold_data,
            noise=noise_data)

def gauss(x, A, mean, sigma):
    return  A * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o","--output_file", help="The output PDF.")
    parser.add_argument("-i","--input_file", nargs="+",
                        help="The _threshold_scan_interpreted_scurve.npz file(s).")
    parser.add_argument("-totfit","--totfit", action="store_true")
    parser.add_argument("-thfit","--thfit", action="store_true")
    parser.add_argument("-thi","--thfile", help="The _threshold.npz file(s).")
    args = parser.parse_args()

    files = []
    #print(args.input_file)
    if args.input_file[0]=="all":
        files.extend(glob.glob("*_threshold_scan_interpreted_scurve.npz"))
    else:
        for pattern in args.input_file:
            files.extend(glob.glob(pattern, recursive=True))
    # if args.input_file:  # If anything was given on the command line
    #     for pattern in args.input_file:
    #         files.extend(glob.glob(pattern, recursive=True))
    # else:
    #     #files.extend(glob.glob("output_data/module_0/chip_0/*_threshold_scan_interpreted_scurve.npz"))
    #     files.extend(glob.glob("*_threshold_scan_interpreted_scurve.npz"))

    files.sort()


    # Load results from NPZ files
    thresholds = np.full((512, 512), np.nan)
    noise = np.full((512, 512), np.nan)
    tot = np.zeros((198,128))
    occupancy_npz = np.full((512,512,198),np.nan)


    for i,fp in enumerate(tqdm(files, unit="file")):
        with np.load(fp) as data:
            charge_dac_edges = np.full((len(data["charge_edges"])),np.nan)

            overwritten = (~np.isnan(thresholds)) & (~np.isnan(data['thresholds']))
            n_overwritten = np.count_nonzero(overwritten)
            if n_overwritten:
                print("WARNING Multiple values of threshold for the same pixel(s)")
                print(f"    count={n_overwritten}, file={fp}")
            thresholds = np.where(np.isnan(thresholds), data['thresholds'], thresholds)

            overwritten = (~np.isnan(noise)) & (~np.isnan(data['noise']))
            n_overwritten = np.count_nonzero(overwritten)
            if n_overwritten:
                print("WARNING Multiple values of threshold for the same pixel(s)")
                print(f"    count={n_overwritten}, file={fp}")
            noise = np.where(np.isnan(noise), data['noise'], noise)

            overwritten = (~np.isnan(charge_dac_edges)) & (~np.isnan(data['charge_edges']))
            n_overwritten = np.count_nonzero(overwritten)
            if n_overwritten:
                print("WARNING Multiple values of threshold for the same pixel(s)")
                print(f"    count={n_overwritten}, file={fp}")
            charge_dac_edges = np.where(np.isnan(charge_dac_edges), data['charge_edges'], charge_dac_edges)

            col = data['col']
            tot+= data["tot"]

            overwritten = (~np.isnan(occupancy_npz)) & (~np.isnan(data['occup']))
            n_overwritten = np.count_nonzero(overwritten)
            if n_overwritten:
                print("WARNING Multiple values of threshold for the same pixel(s)")
                print(f"    count={n_overwritten}, file={fp}")
            occupancy_npz = np.where(np.isnan(occupancy_npz), data['occup'], occupancy_npz)


    np.savez_compressed(
        "all_thresholds_casc.npz",
        all_th = thresholds,
        all_noise = noise,
        all_tot = tot,
        all_occup = occupancy_npz)
    print("\"*.npz\" file is created.")


    global mean_b
    mean_b =  np.zeros((4, 1), dtype="float")

    # Do the plotting
    with PdfPages(args.output_file) as pdf:
        plt.figure(figsize=(6.4, 4.8))

        plt.annotate(
            split_long_text(
                "This file was generated by joining the following\n\n"
                + "\n".join(files)
                ), (0.5, 0.5), ha='center', va='center')
        plt.gca().set_axis_off()
        pdf.savefig(); plt.clf()

        # Threshold hist
        m1 = int(max(0, thresholds.min() - 2))
        m2 = int(min(100, thresholds.max() + 2))

        for i, (fc, lc, name) in enumerate(FRONTENDS):
            # if fc >= col_stop[i] or lc < col_start[i]:
            #     continue
            th = thresholds[fc:lc+1,:]
            th_mean = ufloat(np.mean(th[th>0]), np.std(th[th>0], ddof=1))
            th_m = np.mean(th[th>0])
            bin_height, bin_edge, _ = plt.hist(th.reshape(-1), bins=m2-m1, range=[m1, m2],
                     label=f"{name}", histtype='step', color=f"C{i}")
            entries = np.sum(bin_height)

        plt.suptitle("Threshold distribution")
        plt.xlabel("Threshold [DAC]")
        plt.ylabel("Pixels / bin")
        set_integer_ticks(plt.gca().yaxis)
        plt.legend(loc="upper left", fontsize=9)
        plt.grid(axis='y')
        pdf.savefig();plt.clf()

        #############################################
        #         FIT THRESHOLD DISTRIBUTION        #
        #############################################

        for i, (fc, lc, name) in enumerate(FRONTENDS):
            # if fc >= col_stop[i] or lc < col_start[i]:
            #     continue
            th = thresholds[fc:lc+1,:]
            th_mean = ufloat(np.mean(th[th>0]), np.std(th[th>0], ddof=1))
            th_m = np.mean(th[th>0])
            bin_height, bin_edge, _ = plt.hist(th.reshape(-1), bins=m2-m1, range=[m1, m2],
                     label=f"{name}", histtype='step', color=f"C{i}")
            entries = np.sum(bin_height)

            if entries!=0:
                plt.clf()

                bin_height, bin_edge, _ = plt.hist(th.reshape(-1), bins=m2-m1, range=[m1, m2],
                         label=f"{name}", histtype='step', color=f"C{i}")

                #Fit
                low = [0.1*entries,10, 0]
                up = [0.3*entries+1, 200, 30]


                bin_center = (bin_edge[:-1]+bin_edge[1:])/2
                popt, pcov = curve_fit(gauss, bin_center, bin_height,
                    p0 = [0.2*entries, th_m, 3], bounds=(low, up))
                mean_b[i] = popt[1]
                perr = np.sqrt(np.diag(pcov))

                # Print threshold output
                print("\nTHRESHOLD:\n")
                print(f"popt = {popt}")
                print(f"perr = {perr}")

                xb = np.arange(bin_edge[0], bin_edge[-1], 0.005)
                plt.plot(xb, gauss(xb, *popt), "r-", label=f"fit {name}:"
                    f"\n$mean ={ufloat(popt[1], perr[1]):L}$"
                    f"\n$sigma ={ufloat(popt[2],perr[2]):L}$")


                #Save results in a txt file
                with open(f"th_fitresults[all {name}].txt", "w") as outf:
                    print("#A#mean#sigma:", file=outf)
                    print(*popt, file=outf)
                    print("#SA#Smean#Ssigma:", file=outf)
                    print(*perr, file=outf)

                plt.suptitle("Threshold distribution")
                plt.xlabel("Threshold [DAC]")
                plt.ylabel("Pixels / bin")
                set_integer_ticks(plt.gca().yaxis)
                plt.legend(loc="upper left", fontsize=9)
                plt.grid(axis='y')
                pdf.savefig();

                plt.xlim([60,120])
                pdf.savefig()
                plt.savefig(f"all_casc_thdist_200.png")
                plt.clf()


        plt.clf()


        # Threshold map
        plt.axes((0.125, 0.11, 0.775, 0.72))
        edges = np.linspace(0, 512, 513, endpoint=True)
        plt.pcolormesh(edges, edges, thresholds.transpose(),
                       rasterized=True)  # Necessary for quick save and view in PDF
        plt.suptitle("Threshold map")
        plt.xlabel("Column")
        plt.ylabel("Row")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = plt.colorbar()
        cb.set_label("Threshold [DAC]")
        frontend_names_on_top()
        pdf.savefig(); plt.clf()

        # Noise hist
        m = int(np.ceil(noise.max(initial=0, where=np.isfinite(noise)))) + 1
        for i, (fc, lc, name) in enumerate(FRONTENDS):
            ns = noise[fc:lc+1,:]
            noise_mean = ufloat(np.mean(ns[ns>0]), np.std(ns[ns>0], ddof=1))
            plt.hist(ns.reshape(-1), bins=min(20*m, 100), range=[0, m],
                    label=f"{name} ${noise_mean:L}$", histtype='step', color=f"C{i}")
        plt.suptitle(f"Noise (width of s-curve slope) distribution")
        plt.xlabel("Noise [DAC]")
        plt.ylabel("Pixels / bin")
        set_integer_ticks(plt.gca().yaxis)
        plt.grid(axis='y')
        plt.legend()
        pdf.savefig(); plt.clf()

        # Noise map
        plt.axes((0.125, 0.11, 0.775, 0.72))
        plt.pcolormesh(edges, edges, noise.transpose(),
                       rasterized=True)  # Necessary for quick save and view in PDF
        plt.suptitle("Noise (width of s-curve slope) map")
        plt.xlabel("Column")
        plt.ylabel("Row")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = plt.colorbar()
        cb.set_label("Noise [DAC]")
        #frontend_names_on_top()
        pdf.savefig(); plt.clf()




        # ToT vs injected charge as 2D histogram
        #for i, (fc, lc, name) in enumerate(FRONTENDS):
        #print(np.nonzero(tot))
        plt.pcolormesh(
            charge_dac_edges[:-1], np.linspace(-0.5, 127.5, 128, endpoint=True),
            tot.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF
        plt.suptitle(f"ToT curve (Cascode)")
        plt.xlabel("Injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = integer_ticks_colorbar()
        cb.set_label("Hits / bin")
        pdf.savefig(); plt.clf()

        # ToT vs injected charge as 2D histogram SHIFTED
        charge_shifted = charge_dac_edges[:-1] - 29
        plt.pcolormesh(
            charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
            tot.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

        plt.suptitle(f"ToT curve (Cascode)")
        plt.xlabel("Injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = integer_ticks_colorbar()
        cb.set_label("Hits / bin")
        plt.xlim([0,200])
        pdf.savefig(); plt.clf()

        if args.totfit:


            ##########################################################
            #                   CLEAR DATA
            ##########################################################

            # ToT occu > 100
            tot_100 = tot
            tot_100[tot_100<100] = 0

            # ToT vs injected charge as 2D histogram SHIFTED
            charge_shifted = charge_dac_edges[:-1] - 29
            plt.pcolormesh(
                charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                tot_100.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

            plt.title("Hits/bin > 100")
            plt.suptitle(f"ToT curve (Cascode)")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            plt.xlim([0,200])
            pdf.savefig(); plt.clf()

            # ToT occu > 200
            tot_200 = tot
            tot_200[tot_200<200] = 0

            # ToT vs injected charge as 2D histogram SHIFTED
            charge_shifted = charge_dac_edges[:-1] - 29
            plt.pcolormesh(
                charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                tot_200.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

            plt.title("Hits/bin > 200")
            plt.suptitle(f"ToT curve (Cascode)")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            plt.xlim([0,200])
            pdf.savefig(); plt.clf()

            # ToT occu > 250
            tot_250 = tot
            tot_250[tot_250<250] = 0

            # ToT vs injected charge as 2D histogram SHIFTED
            charge_shifted = charge_dac_edges[:-1] - 29
            plt.pcolormesh(
                charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                tot_250.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

            plt.title("Hits/bin > 250")
            plt.suptitle(f"ToT curve (Cascode)")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            plt.xlim([0,200])
            pdf.savefig(); plt.clf()

            # ToT occu > 300
            tot_300 = tot
            tot_300[tot_300<300] = 0

            # ToT vs injected charge as 2D histogram SHIFTED
            charge_shifted = charge_dac_edges[:-1] - 29
            plt.pcolormesh(
                charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                tot_300.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

            plt.title("Hits/bin > 300")
            plt.suptitle(f"ToT curve (Cascode)")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            plt.xlim([0,200])
            pdf.savefig(); plt.clf()


            ########################################################
            #                ToT vs injected charge
            ########################################################

            # FIRST (NO CUT):
            #   Mean of ToT for each value of Injected charge
            tot_temp = np.tile(np.linspace(0, 127, 128, endpoint=True), (198,1))
            tot_mean= np.sum(tot_temp*tot,axis=1)/ np.sum(tot, axis=1)
            del tot_temp

            # PLOT
            plt.plot(charge_dac_edges[:-1], tot_mean, rasterized=True)
            plt.title("Mean of ToT for each value of injected charge", fontsize=9)
            plt.suptitle(f"ToT curve (Cascode)")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            plt.ylim([0,128])
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            pdf.savefig(); plt.clf()

            # FIRST (CUT >200):
            #   Mean of ToT for each value of Injected charge
            tot_temp = np.tile(np.linspace(0, 127, 128, endpoint=True), (198,1))
            tot_mean_200= np.sum(tot_temp*tot_200,axis=1)/ np.sum(tot_200, axis=1)
            del tot_temp

            # PLOT
            # plt.plot(charge_dac_edges[:-1], tot_mean_200, rasterized=True)
            # plt.title("Mean of ToT for each value of injected charge (hits/bin>200)",
            #     fontsize=9)
            # plt.suptitle(f"ToT curve (Cascode)")
            # plt.xlabel("Injected charge [DAC]")
            # plt.ylabel("ToT [25 ns]")
            # plt.ylim([0,128])
            # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            # pdf.savefig(); plt.clf()

            # FIRST (CUT >300):
            #   Mean of ToT for each value of Injected charge
            tot_temp = np.tile(np.linspace(0, 127, 128, endpoint=True), (198,1))
            tot_mean_300= np.sum(tot_temp*tot_300,axis=1)/ np.sum(tot_300, axis=1)
            del tot_temp

            # PLOT
            # plt.plot(charge_dac_edges[:-1], tot_mean_300, rasterized=True)
            # plt.title("Mean of ToT for each value of injected charge (hits/bin>300)",
            #     fontsize=9)
            # plt.suptitle(f"ToT curve (Cascode)")
            # plt.xlabel("Injected charge [DAC]")
            # plt.ylabel("ToT [25 ns]")
            # plt.ylim([0,128])
            # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            # pdf.savefig(); plt.clf()


            # SECOND (NO CUT):
            #   Most populated bin of ToT for each value of Injected charge

            most_pop = np.argmax(tot, axis=1)

            #PLOT
            plt.plot(charge_dac_edges[:-1], most_pop, "-", rasterized=True)
            plt.title("Most populated bin of ToT for each value of Injected charge",
                fontsize=9)
            plt.suptitle(f"ToT curve (Cascode) ")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            plt.ylim([0,128])
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            pdf.savefig(); plt.clf()


            # SECOND (CUT>200):
            #   Most populated bin of ToT for each value of Injected charge

            most_pop_200 = np.argmax(tot_200, axis=1)

            #PLOT
            # plt.plot(charge_dac_edges[:-1], most_pop_200, "-", rasterized=True)
            # plt.title("Most populated bin of ToT for each value of Injected charge (hits/bin>200)",
            #     fontsize=9)
            # plt.suptitle(f"ToT curve (Cascode) ")
            # plt.xlabel("Injected charge [DAC]")
            # plt.ylabel("ToT [25 ns]")
            # plt.ylim([0,128])
            # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            # pdf.savefig(); plt.clf()


            # SECOND (CUT>300):
            #   Most populated bin of ToT for each value of Injected charge

            most_pop_300 = np.argmax(tot_300, axis=1)

            #PLOT
            # plt.plot(charge_dac_edges[:-1], most_pop_300, "-", rasterized=True)
            # plt.title("Most populated bin of ToT for each value of Injected charge (hits/bin>300)",
            #     fontsize=9)
            # plt.suptitle(f"ToT curve (Cascode) ")
            # plt.xlabel("Injected charge [DAC]")
            # plt.ylabel("ToT [25 ns]")
            # plt.ylim([0,128])
            # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            # pdf.savefig(); plt.clf()

            # THIRD:
            #   Mean of charge for each value of ToT

            tot_temp = np.tile(np.linspace(0, 197, 198, endpoint=True), (128,1))
            char_tot = tot_temp*tot.transpose()
            c_mean= np.sum(char_tot,axis=1)/ np.sum(tot, axis=0)
            del tot_temp

            # SHIFT ON CHARGE
            c_mean_sh = c_mean-29

            # PLOT W/O SHIFT
            # plt.plot(c_mean, np.linspace(0, 127, 128, endpoint=True), rasterized=True)
            # plt.suptitle(f"ToT curve mean on charge (Cascode)")
            # plt.xlabel("Injected charge [DAC]")
            # plt.ylabel("ToT [25 ns]")
            # plt.ylim([0,128])
            # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            # plt.savefig("mean_on_charge.png")
            # pdf.savefig(); plt.clf()

            # PLOT WITH SHIFT
            plt.plot(c_mean_sh, np.linspace(0, 127, 128, endpoint=True), rasterized=True)
            plt.suptitle(f"ToT curve (Cascode)")
            plt.title("Mean of charge for each value of ToT", fontsize=9)
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            plt.ylim([0,128])
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            #plt.savefig("mean_on_charge_shift.png")
            pdf.savefig(); plt.clf()

            # NO VALUES WITH CHARGE APPROX > 160
            charge_fit = c_mean_sh
            charge_fit[21:] = np.nan

            plt.plot(charge_fit, np.linspace(0, 127, 128, endpoint=True), rasterized=True)
            plt.suptitle(f"ToT curve mean on charge (Cascode)")
            plt.title("Mean of charge for each value of ToT, cut", fontsize=9)
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            plt.ylim([0,128])
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            #plt.savefig("mean_on_charge_shift_clear.png")
            pdf.savefig(); plt.clf()


            # FIT FUNCTIONS

            # Threshold shifted
            th_s = mean_b[1]-29

            # MSE
            def mse(func, x, y, coefs):
                return np.mean((func(x, *coefs) - y)**2)

            ######################################
            #           NO CONSTRAINT
            ######################################

            # CUT x<t (Tot vs charge)
            def func_norm_cut_t(x,a,b,c,t):
                return np.where(x<t, 0, np.maximum(0, a*x+b-(c/(x-t))))

            # CUT x<mean (Tot vs charge)
            def func_norm_cut_mean(x,a,b,c,t):
                return np.where(x<th_s, 0, np.maximum(0, a*x+b-(c/(x-t))))


            ######################################
            #           CONSTRAINT ON A
            ######################################
            # a = (c)/(th_s*(th_s-t)) - (b)/(th_s)
            def func_norm_a(x,b,c,t):
                return np.where(x<th_s, 0, np.maximum(0, (c/(th_s*(th_s-t)) - b/(th_s))*x+b-(c/(x-t))))


            ######################################
            #           CONSTRAINT ON B
            ######################################
            # b = (c)/(th_s-t) - a*th_s
            def func_norm_b(x,a,c,t):
                return np.where(x<th_s, 0, np.maximum(0, a*x+(c/(th_s-t) - a*th_s)-(c/(x-t))))


            ######################################
            #           CONSTRAINT ON C
            ######################################
            # c = (th_s-t)*(a*th_s + b)
            # def func_norm_c2(x,a,b,t):
            #     return np.where(x<th_s, 0, np.maximum(0, a*x+b-(((th_s**2)*a-(a*t*th_s)+(b*th_s)-(t*b))/(x-t))))

            def func_norm_c(x,a,b,t):
                c = (th_s-t)*(a*th_s + b)
                return np.where(x<th_s, 0, np.maximum(0, a*x+b-(c/(x-t))))

            def func_norm_c_inv(x,a,b,t):
                c = (th_s-t)*(a*th_s + b)
                y = (t/2)-(b/(2*a))+(x/(2*a)) + np.sqrt(((t/2)+(b/(2*a))-(x/(2*a)))**2 + (c/a))
                return np.where(y<th_s-20, 0, y)


            ######################################
            #           CONSTRAINT ON T
            ######################################
            # t = th_s - (c)/(a*th_s + b)
            def func_norm_t(x,a,b,c):
                t = th_s - (c)/(a*th_s + b)
                return np.where(x<th_s, 0, np.maximum(0, a*x+b-(c/(x-t))))



            ########################################################
            #               FIT ToT vs injected charge
            ########################################################

            # NO CONSTRAINT
            print("\nFIT NO CONSTRAINT:\n")


            # FIRST (NO CUT):
            #   Mean of ToT for each value of Injected charge

            tot_mean_sh = tot_mean
            mask_tot = np.isfinite(tot_mean_sh)
            tot_mean_sh = tot_mean_sh[mask_tot]
            ch_tot_mean_sh = charge_shifted
            ch_tot_mean_sh = ch_tot_mean_sh[mask_tot]
            #charge_dac_bins2 = occu[mask_tot]


            # FIT
            popt, pcov = curve_fit(func_norm_cut_t, ch_tot_mean_sh, tot_mean_sh,
                p0 = [0.15, 2, 45, 44],bounds=([0 , -100, 0, -40], [0.3, 1000,100, 80]))
                #maxfev=10000)
            perr = np.sqrt(np.diag(pcov))

            # PRINT RESULTS
            print("FIT TOT MEAN (ToT vs charge):")
            print(f"popt = {popt}")
            print(f"perr = {perr}")

            # PLOT
            plt.pcolormesh(
                charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                tot.transpose(), vmin=100, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

            y = np.arange(40, 189, 1)
            #y = np.arange(th_s-0.01, 189, 1)
            plt.plot(y, func_norm_cut_t(y, *popt), "r-", label=f"fit Cascode:\n$a ={ufloat(popt[0],perr[0]):L}$\n"
                f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(popt[2],perr[2]):L}$\n$t = {ufloat(popt[3],perr[3]):L}$")

            plt.xlim([0, 250])
            plt.ylim([0, 60])
            plt.suptitle(f"ToT curve (Cascode)")
            plt.title("Fit no constraints: Mean ToT for injected charge", fontsize=9)
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            plt.legend(loc="upper left")
            pdf.savefig();
            plt.savefig("Totfitcasc(200)_nocos_totmean.png"); plt.clf()

            print(f"COVARIANCE:\n {pcov}")


            # FIRST (CUT >200):
            #   Mean of ToT for each value of Injected charge
            #
            tot_mean_sh = tot_mean_200
            mask_tot = np.isfinite(tot_mean_sh)
            tot_mean_sh = tot_mean_sh[mask_tot]
            ch_tot_mean_sh = charge_shifted
            ch_tot_mean_sh = ch_tot_mean_sh[mask_tot]


            # # FIT
            # popt, pcov = curve_fit(func_norm_cut_mean, ch_tot_mean_sh, tot_mean_sh,
            #     p0 = [0.15, 2, 45, 44],bounds=([0 , -100, 0, -40], [0.3, 100,1000, 80]))
            #     #maxfev=10000)
            # perr = np.sqrt(np.diag(pcov))
            #
            # # PRINT RESULTS
            # print("FIT TOT MEAN (ToT vs charge):")
            # print(f"popt = {popt}")
            # print(f"perr = {perr}")
            #
            # # PLOT
            # plt.pcolormesh(
            #     charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
            #     tot.transpose(), vmin=100, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF
            #
            # y = np.arange(40, 189, 1)
            # #y = np.arange(th_s-0.01, 189, 1)
            # plt.plot(y, func_norm_cut_mean(y, *popt), "r-", label=f"fit:\n$a ={ufloat(popt[0],perr[0]):L}$\n"
            #     f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(popt[2],perr[2]):L}$\n$t = {ufloat(popt[3],perr[3]):L}$")
            #
            # plt.xlim([0, 250])
            # plt.ylim([0, 60])
            # plt.suptitle(f"ToT curve (Cascode)")
            # plt.title("Fit no constraints: Mean ToT for injected charge (CUT)",
            #     fontsize=9)
            # plt.xlabel("Injected charge [DAC]")
            # plt.ylabel("ToT [25 ns]")
            # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            # cb = integer_ticks_colorbar()
            # cb.set_label("Hits / bin")
            # plt.legend(loc="upper left")
            # pdf.savefig();
            # plt.savefig("Totfitcasc(200)_nocos_totmean_cut200.png"); plt.clf()
            #
            # print(f"COVARIANCE:\n {pcov}")


            # SECOND (NO CUT):
            #   Most populated bin of ToT for each value of Injected charge

            tot_most_sh = most_pop
            mask_tot_most = np.isfinite(tot_most_sh)
            tot_most_sh = tot_most_sh[mask_tot_most]
            ch_tot_most_sh = charge_shifted
            ch_tot_most_sh = ch_tot_most_sh[mask_tot_most]

            # FIT
            popt, pcov = curve_fit(func_norm_cut_mean, ch_tot_most_sh, tot_most_sh,
                p0 = [0.15, 2, 44, 45],bounds=([0 , -100, 0, -40], [0.3, 100,1000, 80]))
                #maxfev=10000)
            perr = np.sqrt(np.diag(pcov))

            # PRINT RESULTS
            print("\nFIT TOT MOST POPULATED (ToT vs charge):")
            print(f"popt = {popt}")
            print(f"perr = {perr}")

            # PLOT
            plt.pcolormesh(
                charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                tot.transpose(), vmin=100, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

            y = np.arange(40, 189, 1)
            #y = np.arange(th_s-0.01, 189, 1)
            plt.plot(y, func_norm_cut_mean(y, *popt), "r-", label=f"fit Cascode:\n$a ={ufloat(popt[0],perr[0]):L}$\n"
                f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(popt[2],perr[2]):L}$\n$t = {ufloat(popt[3],perr[3]):L}$")

            plt.xlim([0, 250])
            plt.ylim([0, 60])
            plt.suptitle(f"ToT curve (Cascode)")
            plt.title("Fit no constraints: Most pop ToT for injected charge", fontsize=9)
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            plt.legend(loc="upper left")
            pdf.savefig();
            plt.savefig("Totfitcasc(200)_nocos_totmost.png"); plt.clf()

            print(f"COVARIANCE:\n {pcov}")

            # THIRD:
            #   Mean of charge for each value of ToT

            mask_charge = np.isfinite(charge_fit)
            charge_fit = charge_fit[mask_charge]
            tot_fit = np.linspace(-0.5, 127.5, 128, endpoint=True)
            tot_fit = tot_fit[mask_charge]

            # FIT
            popt, pcov = curve_fit(func_norm_cut_mean, charge_fit, tot_fit, p0 = [0.15, 2, 23, 50])
            perr = np.sqrt(np.diag(pcov))

            print("\nFIT CHARGE MEAN (ToT vs charge):")
            print(f"popt = {popt}")
            print(f"perr = {perr}")

            plt.pcolormesh(
                charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                tot.transpose(), vmin=100, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)


            y = np.arange(charge_shifted[0]-0.1, 189, 1)
            plt.plot(y, func_norm_cut_mean(y, *popt), "r-", label=f"fit Cascode:\n$a ={ufloat(popt[0], perr[0]):L}$\n"
                f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(popt[2],perr[2]):L}$\n$t = {ufloat(popt[3], perr[3]):L}$")

            plt.xlim([0, 250])
            plt.ylim([0, 60])
            plt.suptitle(f"ToT curve (Cascode)")
            plt.title("Fit no constraints: Mean of charge for each ToT", fontsize=9)
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            plt.legend(loc="upper left")
            pdf.savefig();
            plt.savefig("Totfitcasc(200)_nocos_chmean.png"); plt.clf()

            print(f"COVARIANCE:\n {pcov}")


            ########################################################
            #               Injected charge vs ToT
            ########################################################

            # PLOT CHARGE VS TOT
            plt.pcolormesh(
                np.linspace(-0.5, 127.5, 128, endpoint=True), charge_shifted,
                tot, vmin=100, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

            plt.ylim([0, 250])
            plt.xlim([0, 60])
            plt.suptitle(f"ToT curve (Cascode)")
            plt.ylabel("Injected charge [DAC]")
            plt.xlabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            pdf.savefig();
            plt.savefig("Totcasc(200)_inv.png"); plt.clf()


            # FIRST (NO CUT):
            #   Mean of ToT for each value of Injected charge

            # PLOT
            tot_mean_sh[0:3]=0
            plt.plot(tot_mean_sh, ch_tot_mean_sh, rasterized=True)
            plt.title("Mean of ToT for each value of injected charge", fontsize=9)
            plt.suptitle(f"ToT curve (Cascode)")
            plt.ylabel("Injected charge [DAC]")
            plt.xlabel("ToT [25 ns]")
            plt.ylim([0,200])
            plt.xlim([0,128])
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            pdf.savefig(); plt.clf()


            # THIRD:
            #   Mean of charge for each value of ToT

            # PLOT
            plt.plot(tot_fit, charge_fit,  rasterized=True)
            plt.suptitle(f"ToT curve (Cascode)")
            plt.title("Mean of charge for each value of ToT")
            plt.ylabel("Injected charge [DAC]")
            plt.xlabel("ToT [25 ns]")
            plt.ylim([0,200])
            plt.xlim([0,128])
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            #plt.savefig("mean_on_charge_shift_inv.png")
            pdf.savefig(); plt.clf()



            ########################################################
            #               FIT Injected charge vs ToT
            ########################################################

            # FIT FUNCTIONS
            def func_norm_inv(x,a,b,c,t):
                y = (t/2)-(b/(2*a))+(x/(2*a)) + np.sqrt(((t/2)+(b/(2*a))-(x/(2*a)))**2 + (c/a))
                return np.where(y<th_s-20, 0, y)
                #return np.where(y<charge_fit[0], 0, y)

            # def func_norm_inv_cut_t(x,a,b,c,t):
            #     y = (t/2)-(b/(2*a))+(x/(2*a)) + np.sqrt(((t/2)+(b/(2*a))-(x/(2*a)))**2 + (c/a))
            #     return np.where(y<t, 0, y)

            def func_norm_inv_cut_ch(x,a,b,c,t):
                y = (t/2)-(b/(2*a))+(x/(2*a)) + np.sqrt(((t/2)+(b/(2*a))-(x/(2*a)))**2 + (c/a))
                return np.where(y<charge_fit[0], 0, y)


            # FIRST (NO CUT):
            #   Mean of ToT for each value of Injected charge
            popt, pcov = curve_fit(func_norm_inv, tot_mean_sh, ch_tot_mean_sh,
                p0 = [0.13, -0.8, 44, 45])
            perr = np.sqrt(np.diag(pcov))

            print("\nFIT INVERSION TOT MEAN (Charge vs ToT):")
            print(f"popt = {popt}")
            print(f"perr = {perr}")

            plt.pcolormesh(
                np.linspace(-0.5, 127.5, 128, endpoint=True), charge_shifted,
                tot, vmin=100, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

            y = np.linspace(-0.5, 127.5, 128, endpoint=True)
            #y = np.arange(-2, 128, 1)
            plt.plot(y, func_norm_inv(y, *popt), "r-", label=f"fit Cascode:\n$a ={ufloat(popt[0], perr[0]):L}$\n"
                f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(popt[2],perr[2]):L}$\n$t = {ufloat(popt[3], perr[3]):L}$")

            plt.ylim([0, 250])
            plt.xlim([0, 60])
            plt.title("Fit no constraints: Mean ToT for injected charge", fontsize=9)
            plt.suptitle(f"ToT curve (Cascode)")
            plt.ylabel("Injected charge [DAC]")
            plt.xlabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            plt.legend(loc="upper right")
            pdf.savefig();
            plt.savefig("Totfitcasc(200)_nocos_totmean_inv.png"); plt.clf()

            print(f"COVARIANCE:\n {pcov}")


            # THIRD:
            #   Mean of charge for each value of ToT


            popt, pcov = curve_fit(func_norm_inv, tot_fit, charge_fit,
                p0 = [0.13, -0.8, 40, 45])
            perr = np.sqrt(np.diag(pcov))

            print("\nFIT CHARGE MEAN (Charge vs ToT):")
            print(f"popt = {popt}")
            print(f"perr = {perr}")

            plt.pcolormesh(
                np.linspace(-0.5, 127.5, 128, endpoint=True), charge_shifted,
                tot, vmin=100, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)


            #y = np.arange(0, 128, 1)
            y=np.linspace(-0.5, 127.5, 128, endpoint=True)
            plt.plot(y, func_norm_inv(y, *popt), "r-", label=f"fit Cascode:\n$a ={ufloat(popt[0], perr[0]):L}$\n"
                f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(popt[2],perr[2]):L}$\n$t = {ufloat(popt[3], perr[3]):L}$")

            plt.ylim([0, 250])
            plt.xlim([0, 60])
            plt.suptitle(f"ToT curve (Cascode)")
            plt.title("Fit no constraints: Mean of charge for each ToT", fontsize=9)
            plt.ylabel("Injected charge [DAC]")
            plt.xlabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            plt.legend(loc="upper right")
            pdf.savefig();
            plt.savefig("Totfitcasc(200)_nocos_chmean_inv.png"); plt.clf()

            print(f"COVARIANCE:\n {pcov}")


            ###################################
            #           CONSTRAINT ON C
            ###################################

            print("CONSTRAINT C\n")

            # FIRST (NO CUT):
            #   Mean of ToT for each value of Injected charge

            # CONSTRAINT ON C
            popt, pcov = curve_fit(func_norm_c, ch_tot_mean_sh, tot_mean_sh,
                p0 = [0.15, -2, 45])
            perr = np.sqrt(np.diag(pcov))

            # PRINT RESULTS
            print("FIT TOT MEAN CONSTRAINT C (ToT vs charge):")
            print(f"popt = {popt}")
            print(f"perr = {perr}")

            plt.pcolormesh(
                charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                tot.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

            #(th_s-t)*(a*th_s + b)
            cc = (th_s-popt[2])*(popt[0]*th_s + popt[1])
            d2cc = ((th_s-popt[2])**2)*((th_s**2)*pcov[0][0] + 2*th_s*pcov[0][1] + pcov[1][1]) - 2*(th_s - popt[2])*(popt[0]*th_s + popt[1])*(th_s*pcov[0][2] + pcov[1][2]) + pcov[2][2]*(popt[0]*th_s + popt[1])**2
            dcc = np.sqrt(d2cc)

            y = np.arange(40, 189, 1)
            plt.plot(y, func_norm_c(y, *popt), "r-", label=f"fit Cascode:\n$a ={ufloat(popt[0],perr[0]):L}$\n"
                f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(cc,dcc):L}$\n$t = {ufloat(popt[2],perr[2]):L}$")

            plt.xlim([0, 250])
            plt.ylim([0, 60])
            plt.title("Fit constraint on c: Mean ToT for injected charge", fontsize=9)
            plt.suptitle(f"ToT curve fit (Cascode)")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            plt.legend(loc="upper left")
            plt.savefig("Totfitcasc(200)_cosc_totmean.png")
            pdf.savefig(); plt.clf()

            print(f"COVARIANCE:\n {pcov}")



            # THIRD:
            #   Mean of charge for each value of ToT

            # CONSTRAINT ON C
            popt, pcov = curve_fit(func_norm_c, charge_fit, tot_fit,
                p0 = [0.15, -2, 45])
            perr = np.sqrt(np.diag(pcov))

            print("\nFIT CHARGE MEAN CONSTRAINT C (ToT vs charge):")
            print(f"popt = {popt}")
            print(f"perr = {perr}")

            plt.pcolormesh(
                charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                tot.transpose(), vmin=100, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

            #(th_s-t)*(a*th_s + b)
            cc = (th_s-popt[2])*(popt[0]*th_s + popt[1])
            d2cc = ((th_s-popt[2])**2)*((th_s**2)*pcov[0][0] + 2*th_s*pcov[0][1] + pcov[1][1]) - 2*(th_s - popt[2])*(popt[0]*th_s + popt[1])*(th_s*pcov[0][2] + pcov[1][2]) + pcov[2][2]*(popt[0]*th_s + popt[1])**2
            dcc = np.sqrt(d2cc)
            #print(f"DELTAC = {dcc}")

            y = np.arange(charge_shifted[0]-0.1, 189, 1)
            plt.plot(y, func_norm_c(y, *popt), "r-", label=f"fit Cascode:\n$a ={ufloat(popt[0],perr[0]):L}$\n"
                f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(cc,dcc):L}$\n$t = {ufloat(popt[2],perr[2]):L}$")

            plt.xlim([0, 250])
            plt.ylim([0, 60])
            plt.title("Fit constraint on c: Mean of charge for each ToT", fontsize=9)
            plt.suptitle(f"ToT curve fit (Cascode)")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            plt.legend(loc="upper left")
            pdf.savefig();
            plt.savefig("Totfitcasc(200)_cosc_chmean.png"); plt.clf()

            print(f"COVARIANCE:\n {pcov}")



            ########################################################
            #               FIT Injected charge vs ToT
            ########################################################

            # FIRST (NO CUT):
            #   Mean of ToT for each value of Injected charge
            popt, pcov = curve_fit(func_norm_c_inv, tot_mean_sh, ch_tot_mean_sh,
                p0 = [0.13, -0.8, 45])
            perr = np.sqrt(np.diag(pcov))

            print("\nFIT INVERSION TOT MEAN (Charge vs ToT):")
            print(f"popt = {popt}")
            print(f"perr = {perr}")

            plt.pcolormesh(
                np.linspace(-0.5, 127.5, 128, endpoint=True), charge_shifted,
                tot, vmin=100, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

            #(th_s-t)*(a*th_s + b)
            cc = (th_s-popt[2])*(popt[0]*th_s + popt[1])
            d2cc = ((th_s-popt[2])**2)*((th_s**2)*pcov[0][0] + 2*th_s*pcov[0][1] + pcov[1][1]) - 2*(th_s - popt[2])*(popt[0]*th_s + popt[1])*(th_s*pcov[0][2] + pcov[1][2]) + pcov[2][2]*(popt[0]*th_s + popt[1])**2
            dcc = np.sqrt(d2cc)

            y = np.linspace(-0.5, 127.5, 128, endpoint=True)
            #y = np.arange(-2, 128, 1)
            plt.plot(y, func_norm_c_inv(y, *popt), "r-", label=f"fit Cascode:\n$a ={ufloat(popt[0],perr[0]):L}$\n"
                f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(cc,dcc):L}$\n$t = {ufloat(popt[2],perr[2]):L}$")

            plt.ylim([0, 250])
            plt.xlim([0, 60])
            plt.title("Fit constraint on c: Mean ToT for injected charge", fontsize=9)
            plt.suptitle(f"ToT curve (Cascode)")
            plt.ylabel("Injected charge [DAC]")
            plt.xlabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            plt.legend(loc="upper right")
            pdf.savefig();
            plt.savefig("Totfitcasc(200)_cosc_totmean_inv.png"); plt.clf()

            print(f"COVARIANCE:\n {pcov}")


            # THIRD:
            #   Mean of charge for each value of ToT

            popt, pcov = curve_fit(func_norm_c_inv, tot_fit, charge_fit,
                p0 = [0.13, -0.8, 45])
            perr = np.sqrt(np.diag(pcov))

            print("\nFIT INVERSION CHARGE MEAN (Charge vs ToT):")
            print(f"popt = {popt}")
            print(f"perr = {perr}")

            plt.pcolormesh(
                np.linspace(-0.5, 127.5, 128, endpoint=True), charge_shifted,
                tot, vmin=100, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

            #(th_s-t)*(a*th_s + b)
            cc = (th_s-popt[2])*(popt[0]*th_s + popt[1])
            d2cc = ((th_s-popt[2])**2)*((th_s**2)*pcov[0][0] + 2*th_s*pcov[0][1] + pcov[1][1]) - 2*(th_s - popt[2])*(popt[0]*th_s + popt[1])*(th_s*pcov[0][2] + pcov[1][2]) + pcov[2][2]*(popt[0]*th_s + popt[1])**2
            dcc = np.sqrt(d2cc)

            #y = np.arange(0, 128, 1)
            y=np.linspace(-0.5, 127.5, 128, endpoint=True)
            plt.plot(y, func_norm_c_inv(y, *popt), "r-", label=f"fit Cascode:\n$a ={ufloat(popt[0],perr[0]):L}$\n"
                f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(cc,dcc):L}$\n$t = {ufloat(popt[2],perr[2]):L}$")

            plt.ylim([0, 250])
            plt.xlim([0, 60])
            plt.suptitle(f"ToT curve (Cascode)")
            plt.title("Fit constraint on c: Mean of charge for each ToT", fontsize=9)
            plt.ylabel("Injected charge [DAC]")
            plt.xlabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            plt.legend(loc="upper right")
            pdf.savefig();
            plt.savefig("Totfitcasc(200)_cosc_chmean_inv.png"); plt.clf()

            print(f"COVARIANCE:\n {pcov}")


        #sys.exit()





        ####################### TOT MEAN VS INJ CHARGE ##########

        # tot_temp = np.tile(np.linspace(0, 127, 128, endpoint=True), (198,1))
        # tot_mean= np.sum(tot_temp*tot,axis=1)/ np.sum(tot, axis=1)
        # del tot_temp
        #
        # plt.plot(charge_dac_edges[:-1], tot_mean, rasterized=True)
        # plt.suptitle(f"ToT curve (Cascode)")
        # plt.xlabel("Injected charge [DAC]")
        # plt.ylabel("ToT [25 ns]")
        # plt.ylim([0,128])
        # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        # pdf.savefig(); plt.clf()


        ########################################TOT SHIFT AND FIT ##############
        # charge_shift = charge_dac_edges[:-1] - 29
        # plt.pcolormesh(
        #     charge_shift, np.linspace(-0.5, 127.5, 128, endpoint=True),
        #     tot.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF
        #
        # plt.suptitle(f"ToT curve (Cascode)")
        # plt.xlabel("True Injected charge [DAC]")
        # plt.ylabel("ToT [25 ns]")
        # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        # cb = integer_ticks_colorbar()
        # cb.set_label("Hits / bin")
        # plt.xlim([0,200])
        # pdf.savefig(); plt.clf()
        #
        # #print(mean_b)
        # def func_norm(x,a,b,c,t):
        #     return np.where(x<mean_b[1]-29, 0, np.maximum(0, a*x+b-(c/(x-t))))
        #
        #
        # tot_mean_shift = tot_mean
        # mask_tot = np.isfinite(tot_mean_shift)
        # tot_mean_shift = tot_mean_shift[mask_tot]
        # occu = charge_shift
        # charge_dac_bins2 = occu[mask_tot]
        #
        # popt, pcov = curve_fit(func_norm, charge_dac_bins2, tot_mean_shift,
        #     p0 = [0.15, 2, 100, -10],bounds=([0 , -100, 0, -40], [0.3, 100,1000, 80]),
        #     maxfev=10000)
        # perr = np.sqrt(np.diag(pcov))
        #
        # print(*popt)
        # print(*perr)
        #
        # plt.pcolormesh(
        #     charge_shift, np.linspace(-0.5, 127.5, 128, endpoint=True),
        #     tot.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF
        #
        # plt.xlim([0, 250])
        # plt.ylim([0, 60])
        # plt.suptitle(f"ToT curve (Cascode)")
        # plt.xlabel("True injected charge [DAC]")
        # plt.ylabel("ToT [25 ns]")
        # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        # cb = integer_ticks_colorbar()
        # cb.set_label("Hits / bin")
        # pdf.savefig(); plt.clf()
        #
        #
        #
        # plt.pcolormesh(
        #     charge_shift, np.linspace(-0.5, 127.5, 128, endpoint=True),
        #     tot.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF
        #
        # y = np.arange(mean_b[1]-29.01, 250, 1)
        # plt.plot(y, func_norm(y, *popt), "r-", label=f"fit Cascode:\na ={ufloat(round(popt[0], 3), round(perr[0], 3))}\nb = {ufloat(round(popt[1],3),round(perr[1],3))}\nc = {ufloat(round(popt[2],3),round(perr[2], 3))}\nt = {ufloat(np.around(popt[3],3),round(perr[3], 3))}")
        # plt.xlim([0, 250])
        # plt.ylim([0, 60])
        #
        # plt.suptitle(f"ToT curve fit (Cascode)")
        # plt.xlabel("True injected charge [DAC]")
        # plt.ylabel("ToT [25 ns]")
        # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        # cb = integer_ticks_colorbar()
        # cb.set_label("Hits / bin")
        # plt.legend(loc="upper left")
        # plt.savefig("Tot_fit_cascode(200).png")
        # pdf.savefig(); plt.clf()





        # S-Curve as 2D histogram
        occupancy_charges = charge_dac_edges.astype(np.float32)
        occupancy_charges = (occupancy_charges[:-1] + occupancy_charges[1:]) / 2
        occupancy_charges = np.tile(occupancy_charges, (512, 512, 1))
        charge_dac_range = [min(charge_dac_edges) - 0.5, max(charge_dac_edges) + 0.5 -1]

        for fc, lc, name in FRONTENDS:
            # if fc >= col_stop or lc < col_start:
            #     continue
            # fc = max(0, fc - col_start)
            # lc = min(col_stop-col_start - 1, lc - col_start)
            #occu= occupancy[fc:lc+1]
            if name=="Cascode":
                plt.hist2d(occupancy_charges[fc:lc+1,:,:].reshape(-1),
                        occupancy_npz[fc:lc+1,:,:].reshape(-1),
                        bins=[198, 150], range=[charge_dac_range, [0, 1.5]],
                        cmin=1, rasterized=True)  # Necessary for quick save and view in PDF
                plt.suptitle(f"S-Curve ({name})")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("Occupancy")
                set_integer_ticks(plt.gca().xaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Pixels / bin")
                plt.savefig("all_casc_thscan_200.png")
                pdf.savefig(); plt.clf()


                # S-curve single pixel and fit
                # 1
                plt.hist2d(occupancy_charges[254,100,:].reshape(-1),
                        occupancy_npz[254,100,:].reshape(-1),
                        bins=[198, 150], range=[charge_dac_range, [0, 1.5]],
                        cmin=1, rasterized=True)  # Necessary for quick save and view in PDF
                plt.suptitle(f"S-Curve (234,100)")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("Occupancy")
                set_integer_ticks(plt.gca().xaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Pixels / bin")
                #plt.savefig("all_norm_thscan_200.png")
                pdf.savefig(); plt.clf()

                def sfit(x, a, b, z, f):
                    return a * erf((x - z)/(np.sqrt(2)*f)) + b

                popt, pcov = curve_fit(sfit, occupancy_charges[254,100,:].reshape(-1),
                    occupancy_npz[254,100,:].reshape(-1))

                print(*popt)

                y=np.arange(0,200,1)
                plt.plot(y, sfit(y,*popt), "-r")

                plt.hist2d(occupancy_charges[254,100,:].reshape(-1),
                        occupancy_npz[254,100,:].reshape(-1),
                        bins=[198, 150], range=[charge_dac_range, [0, 1.5]],
                        cmin=1, rasterized=True)  # Necessary for quick save and view in PDF
                plt.suptitle(f"S-Curve (10,100)")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("Occupancy")
                set_integer_ticks(plt.gca().xaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Pixels / bin")
                #plt.savefig("all_norm_thscan_200.png")
                pdf.savefig(); plt.clf()


                # 2
                plt.hist2d(occupancy_charges[254,400,:].reshape(-1),
                        occupancy_npz[254,400,:].reshape(-1),
                        bins=[198, 150], range=[charge_dac_range, [0, 1.5]],
                        cmin=1, rasterized=True)  # Necessary for quick save and view in PDF
                plt.suptitle(f"S-Curve (10,400)")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("Occupancy")
                set_integer_ticks(plt.gca().xaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Pixels / bin")
                #plt.savefig("all_norm_thscan_200.png")
                pdf.savefig(); plt.clf()


                popt, pcov = curve_fit(sfit, occupancy_charges[254,400,:].reshape(-1),
                    occupancy_npz[254,400,:].reshape(-1))

                print(*popt)

                y=np.arange(0,200,1)
                plt.plot(y, sfit(y,*popt), "-r")

                plt.hist2d(occupancy_charges[254,400,:].reshape(-1),
                        occupancy_npz[254,400,:].reshape(-1),
                        bins=[198, 150], range=[charge_dac_range, [0, 1.5]],
                        cmin=1, rasterized=True)  # Necessary for quick save and view in PDF
                plt.suptitle(f"S-Curve (10,100)")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("Occupancy")
                set_integer_ticks(plt.gca().xaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Pixels / bin")
                #plt.savefig("all_norm_thscan_200.png")
                pdf.savefig(); plt.clf()


                # 3
                plt.hist2d(occupancy_charges[224,225,:].reshape(-1),
                        occupancy_npz[224,225,:].reshape(-1),
                        bins=[198, 150], range=[charge_dac_range, [0, 1.5]],
                        cmin=1, rasterized=True)  # Necessary for quick save and view in PDF
                plt.suptitle(f"S-Curve (0,1)")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("Occupancy")
                set_integer_ticks(plt.gca().xaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Pixels / bin")
                #plt.savefig("all_norm_thscan_200.png")
                pdf.savefig(); plt.clf()

                plt.hist2d(occupancy_charges[224,224+18,:].reshape(-1),
                        occupancy_npz[224,224+18,:].reshape(-1),
                        bins=[198, 150], range=[charge_dac_range, [0, 1.5]],
                        cmin=1, rasterized=True)  # Necessary for quick save and view in PDF
                plt.suptitle(f"S-Curve (0,18)")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("Occupancy")
                set_integer_ticks(plt.gca().xaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Pixels / bin")
                #plt.savefig("all_norm_thscan_200.png")
                pdf.savefig(); plt.clf()

                popt, pcov = curve_fit(sfit, occupancy_charges[224,224+18,:].reshape(-1),
                    occupancy_npz[224,224+18,:].reshape(-1), p0=[0.5, 0.5, 91, 3])

                print(*popt)

                y=np.arange(0,200,1)
                plt.plot(y, sfit(y,*popt), "-r")

                plt.hist2d(occupancy_charges[224,224+18,:].reshape(-1),
                        occupancy_npz[224,224+18,:].reshape(-1),
                        bins=[198, 150], range=[charge_dac_range, [0, 1.5]],
                        cmin=1, rasterized=True)  # Necessary for quick save and view in PDF
                plt.suptitle(f"S-Curve (10,100)")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("Occupancy")
                set_integer_ticks(plt.gca().xaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Pixels / bin")
                #plt.savefig("all_norm_thscan_200.png")
                pdf.savefig(); plt.clf()




                # Fit s-curve of all pixels
                if args.thfit:

                    th_fit = np.zeros((224,512))
                    nois_fit = np.zeros((224,512))

                    # FIT ERROR FUNCTION
                    for i in tqdm(range(0,224), unit="column"):
                        for j in range(0,512):
                            non_zero = np.count_nonzero(occupancy_npz[i+224,j,:])
                            # if non_zero < 5 or (np.isnan(occupancy_npz[i+224,j+224,:])).any():
                            #     #print(f"continue : {i,j , non_zero}")
                            #     continue
                            #print(i,j, np.count_nonzero(occupancy_npz[i+224,j+224,:]))
                            #print(occupancy_npz[0+224,18+224,:])
                            #sys.exit()
                            popt, pcov = curve_fit(sfit, occupancy_charges[i+224,j,:].reshape(-1),
                                occupancy_npz[i+224,j,:].reshape(-1), p0=[0.5, 0.5, 91, 3])

                            th_fit[i,j] = popt[2]
                            nois_fit[i,j] = popt[3]

                    # Save results in "*.npz" file
                    np.savez_compressed(
                        "thr_noise_casc_fit.npz",
                        thr_fit = th_fit,
                        noise_fit = nois_fit)
                    print("\"*.npz\" file is created.")


                    # Threshold map
                    plt.axes((0.125, 0.11, 0.775, 0.72))
                    edges = np.linspace(0, 511, 512, endpoint=True)
                    plt.pcolormesh(edges[224:448], edges, th_fit.transpose(),
                                   rasterized=True)
                    plt.suptitle("Threshold map")
                    plt.xlabel("Column")
                    plt.ylabel("Row")
                    set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                    cb = plt.colorbar()
                    cb.set_label("Threshold [DAC]")
                    #frontend_names_on_top()
                    pdf.savefig(); plt.clf()


                    # Noise map
                    plt.axes((0.125, 0.11, 0.775, 0.72))
                    plt.pcolormesh(edges[224:448], edges, nois_fit.transpose(),
                                   rasterized=True)  # Necessary for quick save and view in PDF
                    plt.suptitle("Noise map")
                    plt.xlabel("Column")
                    plt.ylabel("Row")
                    set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                    cb = plt.colorbar()
                    #plt.clim([1,4])
                    cb.set_label("Noise [DAC]")
                    #frontend_names_on_top()
                    pdf.savefig(); plt.clf()

                else:
                    # Load npz file
                    with np.load(args.thfile) as thdata:
                        th_fit = thdata["thr_fit"]
                        nois_fit = thdata["noise_fit"]


                        # Threshold map
                        plt.axes((0.125, 0.11, 0.775, 0.72))
                        edges = np.linspace(0, 511, 512, endpoint=True)
                        plt.pcolormesh(edges[224:448], edges, th_fit.transpose(),
                                       rasterized=True)
                        plt.suptitle("Threshold map")
                        plt.xlabel("Column")
                        plt.ylabel("Row")
                        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                        cb = plt.colorbar()
                        cb.set_label("Threshold [DAC]")
                        #frontend_names_on_top()
                        pdf.savefig(); plt.clf()


                        # Noise map
                        plt.axes((0.125, 0.11, 0.775, 0.72))
                        plt.pcolormesh(edges[224:448], edges, nois_fit.transpose(),
                                       rasterized=True)  # Necessary for quick save and view in PDF
                        plt.suptitle("Noise map")
                        plt.xlabel("Column")
                        plt.ylabel("Row")
                        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                        cb = plt.colorbar()
                        #plt.clim([1,4])
                        cb.set_label("Noise [DAC]")
                        #frontend_names_on_top()
                        pdf.savefig(); plt.clf()
