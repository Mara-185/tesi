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
from scipy.special import erf
import sys
import copy

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
            pdf.savefig();
            plt.savefig(f"all_{name}_thscan_{the_vh}.png"); plt.clf()


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
            bin_height, bin_edge, _ = plt.hist(th.reshape(-1), bins=m2-m1, range=[m1, m2],
                 label=f"{name}", histtype='step', color=f"C{i}")
##############################################
            bin_center = (bin_edge[:-1]+bin_edge[1:])/2
            if the_vh==140:
                popt, pcov = curve_fit(gauss, bin_center, bin_height,
                    p0 = [2500, 30, 3], bounds=([1000,10, 0],[4000, 150, 30]))
                #low = [1000,10, 0]
                #up = [4000, 150, 30]
            elif the_vh==200:
                popt, pcov = curve_fit(gauss, bin_center, bin_height,
                    p0 = [2000, 60, 5], bounds=([1000,10, 0],[4000, 150, 40]))
            else:
                print("Define new range.")

            print(*popt)
            mean[i] = popt[1]
            perr = np.sqrt(np.diag(pcov))
            print(*perr)
            xb = np.arange(bin_edge[0], bin_edge[-1], 0.005)
            plt.plot(xb, gauss(xb, *popt), "-", label=f"fit {name}:\nmean={ufloat(round(popt[1], 2), round(perr[1],2))}\nsigma={ufloat(round(popt[2], 2), round(perr[2],2))}")
            #Save results in a txt file
            with open(f"th_fitresults_{the_vh}[all {name}].txt", "w") as outf:
                print("#A#mean#sigma:", file=outf)
                print(*popt, file=outf)
                print("#SA#Smean#Ssigma:", file=outf)
                print(*perr, file=outf)

            plt.title(subtitle)
            plt.suptitle(f"Threshold distribution ({name})")
            plt.xlabel("Threshold [DAC]")
            plt.ylabel("Pixels / bin")
            set_integer_ticks(plt.gca().yaxis)
            plt.legend(loc="upper right")
            plt.grid(axis='y')
            pdf.savefig();
            plt.savefig(f"all_{name}_thdist_{the_vh}.png"); plt.clf()

###################################################


        # ToT vs injected charge as 2D histogram
        for i, ((fc, lc, name), hist) in enumerate(zip(FRONTENDS, tot_hist[1:])):
            if fc >= col_stop or lc < col_start:
                continue
            plt.pcolormesh(
                occupancy_edges[2], np.linspace(-0.5, 127.5, 129, endpoint=True),
                hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF
            np.nan_to_num(hist, copy=False)


            plt.title(subtitle)
            plt.suptitle(f"ToT curve ({name})")
            plt.xlabel("Injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            cb = integer_ticks_colorbar()
            cb.set_label("Hits / bin")
            plt.xlim([0,200])
            pdf.savefig(); plt.clf()


            # ToT vs injected charge as 2D histogram SHIFTED
            charge_dac_edges = copy.deepcopy(occupancy_edges[2])
            tot = copy.deepcopy(hist)
            # print("ToT transpose")
            # print(tot.transpose()[0:5, :])
            charge_shifted = copy.deepcopy(charge_dac_edges[:-1]) -29
            plt.pcolormesh(
                charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                tot.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

            plt.suptitle(f"ToT curve ({name})")
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
                tot_100 = copy.deepcopy(tot)
                tot_100[tot_100<100] = 0

                # ToT vs injected charge as 2D histogram SHIFTED
                #charge_shifted = charge_dac_edges[:-1] - 29
                plt.pcolormesh(
                    charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                    tot_100.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

                plt.title("Hits/bin > 100")
                plt.suptitle(f"ToT curve ({name})")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.xlim([0,200])
                pdf.savefig(); plt.clf()

                # ToT occu > 200
                tot_200 = copy.deepcopy(tot)
                tot_200[tot_200<200] = 0

                # ToT vs injected charge as 2D histogram SHIFTED
                #charge_shifted = charge_dac_edges[:-1] - 29
                plt.pcolormesh(
                    charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                    tot_200.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

                plt.title("Hits/bin > 200")
                plt.suptitle(f"ToT curve ({name})")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.xlim([0,200])
                pdf.savefig(); plt.clf()

                # ToT occu > 250
                tot_250 = copy.deepcopy(tot)
                tot_250[tot_250<250] = 0

                # ToT vs injected charge as 2D histogram SHIFTED
                #charge_shifted = charge_dac_edges[:-1] - 29
                plt.pcolormesh(
                    charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                    tot_250.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

                plt.title("Hits/bin > 250")
                plt.suptitle(f"ToT curve ({name})")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.xlim([0,200])
                pdf.savefig(); plt.clf()

                # ToT occu > 300
                tot_300 = copy.deepcopy(tot)
                tot_300[tot_300<300] = 0

                # ToT vs injected charge as 2D histogram SHIFTED
                #charge_shifted = charge_dac_edges[:-1] - 29
                plt.pcolormesh(
                    charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                    tot_300.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

                plt.title("Hits/bin > 300")
                plt.suptitle(f"ToT curve ({name})")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.xlim([0,200])
                pdf.savefig(); plt.clf()


                # ToT occu > 1000
                tot_1000 = copy.deepcopy(tot)
                tot_1000[tot_1000<1000] = 0

                # ToT vs injected charge as 2D histogram SHIFTED
                #charge_shifted = charge_dac_edges[:-1] - 29
                plt.pcolormesh(
                    charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                    tot_1000.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

                plt.title("Hits/bin > 1000")
                plt.suptitle(f"ToT curve ({name})")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.xlim([0,100])
                plt.ylim([0,15])
                pdf.savefig(); plt.clf()


                # ToT occu > 1000
                tot_1000.transpose()[0:5, 45:] = 0

                # ToT vs injected charge as 2D histogram SHIFTED
                #charge_shifted = charge_dac_edges[:-1] - 29
                plt.pcolormesh(
                    charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                    tot_1000.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

                plt.title("Hits/bin > 1000")
                plt.suptitle(f"ToT curve ({name})")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.xlim([0,100])
                plt.ylim([0,15])
                pdf.savefig(); plt.clf()

                # print(tot.transpose()[0, :])
                # print(tot_1000.transpose()[0, :])

                plt.pcolormesh(
                    charge_shifted, np.linspace(0, 127, 128, endpoint=True),
                    tot_1000.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

                plt.title("Hits/bin > 1000")
                plt.suptitle(f"ToT curve ({name})")
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

                tot_temp = np.tile(np.linspace(0, 127, 128, endpoint=True), (charge_dac_bins,1))
                tot_mean= np.sum(tot_temp*tot,axis=1)/ np.sum(tot, axis=1)
                print(tot_temp.shape)
                print(tot.shape)
                del tot_temp

                # PLOT
                #plt.plot(charge_dac_edges[:-1], tot_mean, rasterized=True)
                plt.plot(charge_shifted, tot_mean, rasterized=True)
                plt.title("Mean of ToT for each value of injected charge", fontsize=9)
                plt.suptitle(f"ToT curve ({name})")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                plt.ylim([0,128])
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                pdf.savefig(); plt.clf()

                # FIRST (CUT >200):
                #   Mean of ToT for each value of Injected charge
                # tot_temp = np.tile(np.linspace(0, 127, 128, endpoint=True), (charge_dac_bins,1))
                # tot_mean_200= np.sum(tot_temp*tot_200,axis=1)/ np.sum(tot_200, axis=1)
                # del tot_temp

                # PLOT
                # plt.plot(charge_dac_edges[:-1], tot_mean_200, rasterized=True)
                # plt.title("Mean of ToT for each value of injected charge (hits/bin>200)",
                #     fontsize=9)
                # plt.suptitle(f"ToT curve ({name})")
                # plt.xlabel("Injected charge [DAC]")
                # plt.ylabel("ToT [25 ns]")
                # plt.ylim([0,128])
                # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                # pdf.savefig(); plt.clf()

                # FIRST (CUT >300):
                #   Mean of ToT for each value of Injected charge
                # tot_temp = np.tile(np.linspace(0, 127, 128, endpoint=True), (charge_dac_bins,1))
                # tot_mean_300= np.sum(tot_temp*tot_300,axis=1)/ np.sum(tot_300, axis=1)
                # del tot_temp

                # PLOT
                # plt.plot(charge_dac_edges[:-1], tot_mean_300, rasterized=True)
                # plt.title("Mean of ToT for each value of injected charge (hits/bin>300)",
                #     fontsize=9)
                # plt.suptitle(f"ToT curve ({name})")
                # plt.xlabel("Injected charge [DAC]")
                # plt.ylabel("ToT [25 ns]")
                # plt.ylim([0,128])
                # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                # pdf.savefig(); plt.clf()


                # SECOND (NO CUT):
                #   Most populated bin of ToT for each value of Injected charge

                most_pop = np.argmax(tot, axis=1)

                #PLOT
                #plt.plot(charge_dac_edges[:-1], most_pop, "-", rasterized=True)
                plt.plot(charge_shifted, most_pop, "-", rasterized=True)
                plt.title("Most populated bin of ToT for each value of Injected charge",
                    fontsize=9)
                plt.suptitle(f"ToT curve ({name}) ")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                plt.ylim([0,128])
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                pdf.savefig(); plt.clf()


                # SECOND (CUT>200):
                #   Most populated bin of ToT for each value of Injected charge

                #most_pop_200 = np.argmax(tot_200, axis=1)

                #PLOT
                # plt.plot(charge_dac_edges[:-1], most_pop_200, "-", rasterized=True)
                # plt.title("Most populated bin of ToT for each value of Injected charge (hits/bin>200)",
                #     fontsize=9)
                # plt.suptitle(f"ToT curve ({name}) ")
                # plt.xlabel("Injected charge [DAC]")
                # plt.ylabel("ToT [25 ns]")
                # plt.ylim([0,128])
                # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                # pdf.savefig(); plt.clf()


                # SECOND (CUT>300):
                #   Most populated bin of ToT for each value of Injected charge

                #most_pop_300 = np.argmax(tot_300, axis=1)

                #PLOT
                # plt.plot(charge_dac_edges[:-1], most_pop_300, "-", rasterized=True)
                # plt.title("Most populated bin of ToT for each value of Injected charge (hits/bin>300)",
                #     fontsize=9)
                # plt.suptitle(f"ToT curve ({name}) ")
                # plt.xlabel("Injected charge [DAC]")
                # plt.ylabel("ToT [25 ns]")
                # plt.ylim([0,128])
                # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                # pdf.savefig(); plt.clf()


                # THIRD:
                #   Mean of charge for each value of ToT
                tot_temp = np.tile(np.linspace(0, charge_dac_bins-1, charge_dac_bins, endpoint=True), (128,1))
                char_tot = tot_temp*tot_1000.transpose()
                c_mean= np.sum(char_tot,axis=1)/ np.sum(tot_1000.transpose(), axis=1)
                #c_m = np.sum(char_tot[0, :],axis=1)/ np.sum(tot_1000.transpose()[0, :], axis=1)
                print(tot_temp.shape)
                print(tot_1000.transpose()[0:5,:])
                print(c_mean)
                del tot_temp

                # print("ToT transpose")
                # print(tot.transpose()[0:5, :])
                # print("Tot_300 transpose")
                # print(tot_300.transpose()[0:5, :])
                # print(tot_1000.transpose()[0:5, :])
                # print("char_tot, tot_1000")
                # print(char_tot)
                #print(char_tot.shape)

                # print(c_mean)


                # SHIFT ON CHARGE
                #c_mean_sh = copy.deepcopy(c_mean)-29

                # PLOT W/O SHIFT
                plt.plot(c_mean, np.linspace(0, 127, 128, endpoint=True), rasterized=True)
                plt.suptitle(f"ToT curve mean on charge ({name})")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                plt.ylim([0,128])
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                plt.savefig("mean_on_charge.png")
                pdf.savefig(); plt.clf()


                # PLOT WITH SHIFT
                # plt.plot(c_mean_sh, np.linspace(0, 127, 128, endpoint=True), rasterized=True)
                # plt.suptitle(f"ToT curve ({name})")
                # plt.title("Mean of charge for each value of ToT", fontsize=9)
                # plt.xlabel("Injected charge [DAC]")
                # plt.ylabel("ToT [25 ns]")
                # plt.ylim([0,128])
                # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                # #plt.savefig("mean_on_charge_shift.png")
                # pdf.savefig(); plt.clf()


                # NO VALUES WITH CHARGE APPROX > 160
                charge_fit = copy.deepcopy(c_mean)
                #charge_fit[:3] = np.nan
                charge_fit[48:] = np.nan

                # print(c_mean)
                # print(charge_fit)

                plt.plot(charge_fit, np.linspace(0, 127, 128, endpoint=True), rasterized=True)
                plt.suptitle(f"ToT curve mean on charge ({name})")
                plt.title("Mean of charge for each value of ToT, cut", fontsize=9)
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                #plt.ylim([0,128])
                plt.ylim([0, 60])
                plt.xlim([0, 250])
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                #plt.savefig("mean_on_charge_shift_clear.png")
                pdf.savefig(); plt.clf()


                # CUT 300
                # tot_temp = np.tile(np.linspace(0, charge_dac_bins-1, charge_dac_bins, endpoint=True), (128,1))
                # char_tot_300 = tot_temp*tot_300.transpose()
                # c_mean_300= np.sum(char_tot_300,axis=1)/ np.sum(tot, axis=0)
                # del tot_temp
                # charge_fit_300 = c_mean_300 #_sh
                # charge_fit_300[:3] = np.nan
                # charge_fit_300[48:] = np.nan
                #
                # plt.plot(c_mean_300, np.linspace(0, 127, 128, endpoint=True), rasterized=True)
                # plt.suptitle(f"ToT curve mean on charge ({name}) ")
                # plt.title("Mean of charge for each value of ToT, cut data 300", fontsize=9)
                # plt.xlabel("Injected charge [DAC]")
                # plt.ylabel("ToT [25 ns]")
                # plt.ylim([0,128])
                # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                # #plt.savefig("mean_on_charge_shift_clear.png")
                # pdf.savefig(); plt.clf()


                # FIT FUNCTIONS

                # Threshold shifted
                th_s = mean[i]-29


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
                    return np.where(x<t, 0, np.maximum(0, a*x+b-(c/(x-t))))
                    #return np.where(x<th_s, 0, np.maximum(0, a*x+b-(c/(x-t))))

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

                tot_mean_sh = copy.deepcopy(tot_mean)
                mask_tot = np.isfinite(tot_mean_sh)
                tot_mean_sh = tot_mean_sh[mask_tot]
                ch_tot_mean_sh = copy.deepcopy(charge_shifted)
                ch_tot_mean_sh = ch_tot_mean_sh[mask_tot]


                # FIT
                popt, pcov = curve_fit(func_norm_cut_t, ch_tot_mean_sh, tot_mean_sh,
                    p0 = [0.15, 2, 45, 44],bounds=([0 , -100, 0, -40], [0.3, 1000,300, 80]))
                    #maxfev=10000)
                perr = np.sqrt(np.diag(pcov))

                # PRINT RESULTS
                print("FIT TOT MEAN (ToT vs charge):")
                print(f"popt = {popt}")
                print(f"perr = {perr}")

                # PLOT
                plt.pcolormesh(
                    charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                    tot.transpose(), vmin=200, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

                y = np.arange(20, 189, 1)
                #y = np.arange(th_s-0.01, 189, 1)
                plt.plot(y, func_norm_cut_t(y, *popt), "r-", label=f"fit {name}:\n$a ={ufloat(popt[0],perr[0]):L}$\n"
                    f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(popt[2],perr[2]):L}$\n$t = {ufloat(popt[3],perr[3]):L}$")

                plt.xlim([0, 250])
                plt.ylim([0, 60])
                plt.suptitle(f"ToT curve ({name})")
                plt.title("Fit no constraints: Mean ToT for injected charge", fontsize=9)
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.legend(loc="upper left")
                pdf.savefig();
                plt.savefig(f"Totfit{name}(200)_nocos_totmean.png"); plt.clf()

                print(f"COVARIANCE:\n {pcov}")


                # FIRST (CUT >200):
                #   Mean of ToT for each value of Injected charge
                #
                # tot_mean_sh = tot_mean_200
                # mask_tot = np.isfinite(tot_mean_sh)
                # tot_mean_sh = tot_mean_sh[mask_tot]
                # ch_tot_mean_sh = charge_shifted
                # ch_tot_mean_sh = ch_tot_mean_sh[mask_tot]


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
                # plt.suptitle(f"ToT curve ({name})")
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

                # tot_most_sh = most_pop
                # mask_tot_most = np.isfinite(tot_most_sh)
                # tot_most_sh = tot_most_sh[mask_tot_most]
                # ch_tot_most_sh = charge_shifted
                # ch_tot_most_sh = ch_tot_most_sh[mask_tot_most]

                # FIT
                # popt, pcov = curve_fit(func_norm_cut_mean, ch_tot_most_sh, tot_most_sh,
                #     p0 = [0.15, 2, 44, 45],bounds=([0 , -100, 0, -40], [0.3, 100,1000, 80]))
                #     #maxfev=10000)
                # perr = np.sqrt(np.diag(pcov))
                #
                # # PRINT RESULTS
                # print("\nFIT TOT MOST POPULATED (ToT vs charge):")
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
                # plt.plot(y, func_norm_cut_mean(y, *popt), "r-", label=f"fit {name}:\n$a ={ufloat(popt[0],perr[0]):L}$\n"
                #     f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(popt[2],perr[2]):L}$\n$t = {ufloat(popt[3],perr[3]):L}$")
                #
                # plt.xlim([0, 250])
                # plt.ylim([0, 60])
                # plt.suptitle(f"ToT curve ({name})")
                # plt.title("Fit no constraints: Most pop ToT for injected charge", fontsize=9)
                # plt.xlabel("Injected charge [DAC]")
                # plt.ylabel("ToT [25 ns]")
                # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                # cb = integer_ticks_colorbar()
                # cb.set_label("Hits / bin")
                # plt.legend(loc="upper left")
                # pdf.savefig();
                # plt.savefig(f"Totfit{name}(200)_nocos_totmost.png"); plt.clf()
                #
                # print(f"COVARIANCE:\n {pcov}")

                # THIRD:
                #   Mean of charge for each value of ToT

                mask_charge = np.isfinite(charge_fit)
                charge_fit2 = copy.copy(charge_fit)
                charge_fit2 = charge_fit2[mask_charge]
                tot_fit = np.linspace(0, 127, 128, endpoint=True)
                tot_fit = tot_fit[mask_charge]

                # print("Charge_fit")
                # print(charge_fit)
                # print("mask")
                # print(mask_charge)
                # print("tot")
                # print(tot_fit)

                # FIT
                popt, pcov = curve_fit(func_norm_cut_t, charge_fit2, tot_fit,
                    p0 = [0.26, 4.8, 50, 48], bounds=([0 , -100, 0, -40], [0.3, 1000,300, 80]))
                perr = np.sqrt(np.diag(pcov))

                print("\nFIT CHARGE MEAN (ToT vs charge):")
                print(f"popt = {popt}")
                print(f"perr = {perr}")

                plt.pcolormesh(
                    charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                    tot.transpose(), vmin=200, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)


                y = np.arange(20, 189, 1)
                plt.plot(y, func_norm_cut_t(y, *popt), "r-", label=f"fit {name}:\n$a ={ufloat(popt[0], perr[0]):L}$\n"
                    f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(popt[2],perr[2]):L}$\n$t = {ufloat(popt[3], perr[3]):L}$")

                plt.xlim([0, 250])
                plt.ylim([0, 60])
                plt.suptitle(f"ToT curve ({name})")
                plt.title("Fit no constraints: Mean of charge for each ToT", fontsize=9)
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.legend(loc="upper left")
                pdf.savefig();
                plt.savefig(f"Totfit{name}(200)_nocos_chmean.png"); plt.clf()

                print(f"COVARIANCE:\n {pcov}")

                sys.exit()



                ########################################################
                #               FIT Injected charge vs ToT
                ########################################################

                # FIT FUNCTIONS
                def func_norm_inv(x,a,b,c,t):
                    y = (t/2)-(b/(2*a))+(x/(2*a)) + np.sqrt(((t/2)+(b/(2*a))-(x/(2*a)))**2 + (c/a))
                    #return np.where(y<th_s-20, 0, y)
                    return np.where(y<charge_fit[0], 0, y)

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
                plt.plot(y, func_norm_inv(y, *popt), "r-", label=f"fit {name}:\n$a ={ufloat(popt[0], perr[0]):L}$\n"
                    f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(popt[2],perr[2]):L}$\n$t = {ufloat(popt[3], perr[3]):L}$")

                plt.ylim([0, 250])
                plt.xlim([0, 60])
                plt.title("Fit no constraints: Mean ToT for injected charge", fontsize=9)
                plt.suptitle(f"ToT curve ({name})")
                plt.ylabel("Injected charge [DAC]")
                plt.xlabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.legend(loc="upper right")
                pdf.savefig();
                plt.savefig(f"Totfit{name}(200)_nocos_totmean_inv.png"); plt.clf()

                print(f"COVARIANCE:\n {pcov}")


                # THIRD:
                #   Mean of charge for each value of ToT


                popt, pcov = curve_fit(func_norm_inv, tot_fit, charge_fit2,
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
                plt.plot(y, func_norm_inv(y, *popt), "r-", label=f"fit {name}:\n$a ={ufloat(popt[0], perr[0]):L}$\n"
                    f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(popt[2],perr[2]):L}$\n$t = {ufloat(popt[3], perr[3]):L}$")

                plt.ylim([0, 250])
                plt.xlim([0, 60])
                plt.suptitle(f"ToT curve ({name})")
                plt.title("Fit no constraints: Mean of charge for each ToT", fontsize=9)
                plt.ylabel("Injected charge [DAC]")
                plt.xlabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.legend(loc="upper right")
                pdf.savefig();
                plt.savefig(f"Totfit{name}(200)_nocos_chmean_inv.png"); plt.clf()

                print(f"COVARIANCE:\n {pcov}")


                #sys.exit()

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
                    tot_1000.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

                #(th_s-t)*(a*th_s + b)
                cc = (th_s-popt[2])*(popt[0]*th_s + popt[1])
                d2cc = ((th_s-popt[2])**2)*((th_s**2)*pcov[0][0] + 2*th_s*pcov[0][1] + pcov[1][1]) - 2*(th_s - popt[2])*(popt[0]*th_s + popt[1])*(th_s*pcov[0][2] + pcov[1][2]) + pcov[2][2]*(popt[0]*th_s + popt[1])**2
                dcc = np.sqrt(d2cc)

                y = np.arange(20, 189, 1)
                plt.plot(y, func_norm_c(y, *popt), "r-", label=f"fit {name}:\n$a ={ufloat(popt[0],perr[0]):L}$\n"
                    f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(cc,dcc):L}$\n$t = {ufloat(popt[2],perr[2]):L}$")

                plt.xlim([0, 250])
                plt.ylim([0, 60])
                plt.title("Fit constraint on c: Mean ToT for injected charge", fontsize=9)
                plt.suptitle(f"ToT curve fit ({name})")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.legend(loc="upper left")
                plt.savefig(f"Totfit{name}(200)_cosc_totmean.png")
                pdf.savefig(); plt.clf()

                print(f"COVARIANCE:\n {pcov}")



                # THIRD:
                #   Mean of charge for each value of ToT

                # CONSTRAINT ON C
                popt, pcov = curve_fit(func_norm_c, charge_fit2, tot_fit,
                    p0 = [0.15, -2, 45])
                perr = np.sqrt(np.diag(pcov))

                print("\nFIT CHARGE MEAN CONSTRAINT C (ToT vs charge):")
                print(f"popt = {popt}")
                print(f"perr = {perr}")

                plt.pcolormesh(
                    charge_shifted, np.linspace(-0.5, 127.5, 128, endpoint=True),
                    tot_1000.transpose(), vmin=100, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)

                #(th_s-t)*(a*th_s + b)
                cc = (th_s-popt[2])*(popt[0]*th_s + popt[1])
                d2cc = ((th_s-popt[2])**2)*((th_s**2)*pcov[0][0] + 2*th_s*pcov[0][1] + pcov[1][1]) - 2*(th_s - popt[2])*(popt[0]*th_s + popt[1])*(th_s*pcov[0][2] + pcov[1][2]) + pcov[2][2]*(popt[0]*th_s + popt[1])**2
                dcc = np.sqrt(d2cc)
                #print(f"DELTAC = {dcc}")

                y = np.arange(20, 189, 1)
                plt.plot(y, func_norm_c(y, *popt), "r-", label=f"fit {name}:\n$a ={ufloat(popt[0],perr[0]):L}$\n"
                    f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(cc,dcc):L}$\n$t = {ufloat(popt[2],perr[2]):L}$")

                plt.xlim([0, 250])
                plt.ylim([0, 60])
                plt.title("Fit constraint on c: Mean of charge for each ToT", fontsize=9)
                plt.suptitle(f"ToT curve fit ({name})")
                plt.xlabel("Injected charge [DAC]")
                plt.ylabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.legend(loc="upper left")
                pdf.savefig();
                plt.savefig(f"Totfit{name}(200)_cosc_chmean.png"); plt.clf()

                print(f"COVARIANCE:\n {pcov}")

                sys.exit()


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
                plt.plot(y, func_norm_c_inv(y, *popt), "r-", label=f"fit {name}:\n$a ={ufloat(popt[0],perr[0]):L}$\n"
                    f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(cc,dcc):L}$\n$t = {ufloat(popt[2],perr[2]):L}$")

                plt.ylim([0, 250])
                plt.xlim([0, 60])
                plt.title("Fit constraint on c: Mean ToT for injected charge", fontsize=9)
                plt.suptitle(f"ToT curve ({name})")
                plt.ylabel("Injected charge [DAC]")
                plt.xlabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.legend(loc="upper right")
                pdf.savefig();
                plt.savefig(f"Totfit{name}(200)_cosc_totmean_inv.png"); plt.clf()

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
                plt.plot(y, func_norm_c_inv(y, *popt), "r-", label=f"fit {name}:\n$a ={ufloat(popt[0],perr[0]):L}$\n"
                    f"$b = {ufloat(popt[1],perr[1]):L}$\n$c = {ufloat(cc,dcc):L}$\n$t = {ufloat(popt[2],perr[2]):L}$")

                plt.ylim([0, 250])
                plt.xlim([0, 60])
                plt.suptitle(f"ToT curve ({name})")
                plt.title("Fit constraint on c: Mean of charge for each ToT", fontsize=9)
                plt.ylabel("Injected charge [DAC]")
                plt.xlabel("ToT [25 ns]")
                set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                cb = integer_ticks_colorbar()
                cb.set_label("Hits / bin")
                plt.legend(loc="upper right")
                pdf.savefig();
                plt.savefig(f"Totfit{name}(200)_cosc_chmean_inv.png"); plt.clf()

                print(f"COVARIANCE:\n {pcov}")




                sys.exit()




################################################################################

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
                if the_vh==200:
                    ##############PLOT AND FIT AFTER SHIFT #####################
                    occupancy_shift = occupancy_edges[2]-29
                    plt.pcolormesh(
                        occupancy_shift, np.linspace(-0.5, 127.5, 129, endpoint=True),
                        hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF
                    np.nan_to_num(hist, copy=False)

                    plt.title(subtitle)
                    plt.suptitle(f"ToT curve ({name})")
                    plt.xlabel("True Injected charge [DAC]")
                    plt.ylabel("ToT [25 ns]")
                    set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                    cb = integer_ticks_colorbar()
                    cb.set_label("Hits / bin")
                    pdf.savefig(); plt.clf()

                    print(i, mean[i-1])


                    if i==3:
                        th_140 = 35.34
                    elif i==4:
                        th_140 = 31.70
                    #((th_140)**2)*a-(a*t*th_140)+(b*th_140)-(t*b)

                    print(th_140)

                    def mse(func, x, y, coefs):
                        return np.mean((func(x, *coefs) - y)**2)


                    ######################################
                    #           NO CONSTRAINT
                    ######################################
                    #print(mean_b)
                    def func_norm(x,a,b,c,t):
                        return np.where(x<mean[i-1]-29, 0, np.maximum(0, a*x+b-(c/(x-t))))


                    ######################################
                    #           CONSTRAINT ON C
                    ######################################
                    # (th_140-t)*(a*th_140 + b)
                    def func_norm_c2(x,a,b,t):
                        return np.where(x<mean[i-1]-29, 0, np.maximum(0, a*x+b-((((th_140)**2)*a-(a*t*th_140)+(b*th_140)-(t*b))/(x-t))))

                    def func_norm_c(x,a,b,t):
                        return np.where(x<mean[i-1]-29, 0, np.maximum(0, a*x+b-(((th_140-t)*(a*th_140 + b))/(x-t))))

                    ######################################
                    #           CONSTRAINT ON A
                    ######################################
                    # (c)/(th_140*(th_140-t)) - (b)/(th_140)
                    def func_norm_a(x,b,c,t):
                        return np.where(x<mean[i-1]-29, 0, np.maximum(0, ((c)/(th_140*(th_140-t)) - (b)/(th_140))*x+b-(c/(x-t))))


                    ######################################
                    #           CONSTRAINT ON B
                    ######################################
                    # (c)/(th_140-t) - a*th_140
                    def func_norm_b(x,a,c,t):
                        return np.where(x<mean[i-1]-29, 0, np.maximum(0, a*x+((c)/(th_140-t) - a*th_140)-(c/(x-t))))


                    ######################################
                    #           CONSTRAINT ON T
                    ######################################
                    # th_140 - (c)/(a*th_140 + b)
                    def func_norm_t(x,a,b,c):
                        return np.where(x<mean[i-1]-29, 0, np.maximum(0, a*x+b-(c/(x-(th_140 - (c)/(a*th_140 + b))))))


                    #
                    # def func_{name}2(x,a,b,t):
                    #     return np.where(x<mean[0]-29, 0, np.maximum(0, a*x+b-((((th_140)**2)*a-(a*t*th_140)+(b*th_140)-(t*b))/(x-t))))
                    #
                    #
                    # def func_casc(x,a,b,c,t):
                    #     return np.where(x<mean[i-1]-29, 0, np.maximum(0, a*x+b-(c/(x-t))))

                    tot_mean_shift = tot_mean
                    mask_tot = np.isfinite(tot_mean_shift)
                    tot_mean_shift = tot_mean_shift[mask_tot]
                    occu = occupancy_shift[:-1]
                    charge_dac_bins2 = occu[mask_tot]


                    ######################################
                    #       NO CONSTRAINT
                    #####################################

                    print("NO CONSTRAINT\n")

                    popt, pcov = curve_fit(func_norm, charge_dac_bins2, tot_mean_shift,
                        p0 = [0.15, 2, 100, -10],bounds=([0 , -100, 0, -40], [0.3, 100,1000, 80]),
                        maxfev=10000)

                    perr = np.sqrt(np.diag(pcov))


                    plt.pcolormesh(
                        occupancy_shift, np.linspace(-0.5, 127.5, 129, endpoint=True),
                        hist.transpose(), vmin=300, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

                    print(*popt)
                    print(*perr)

                    y = np.arange(mean[i-1]-29.01, 189, 1)
                    plt.plot(y, func_norm(y, *popt), "r-", label=f"fit:\na ={ufloat(round(popt[0], 3), round(perr[0], 3))}\n"
                        f"b = {ufloat(round(popt[1],3),round(perr[1],3))}\nc = {ufloat(np.around(popt[2],3),round(perr[2], 3))}\nt = {ufloat(np.around(popt[3],3),round(perr[3], 3))}")


                    plt.xlim([0, 250])
                    plt.ylim([0, 60])
                    plt.suptitle(f"ToT curve ({name})")
                    plt.xlabel("True injected charge [DAC]")
                    plt.ylabel("ToT [25 ns]")
                    set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                    cb = integer_ticks_colorbar()
                    cb.set_label("Hits / bin")
                    plt.legend(loc="upper left")
                    pdf.savefig();
                    plt.savefig(f"Tot_fit_{name}(200).png"); plt.clf()

                    mse_n = mse(func_norm, charge_dac_bins2, tot_mean_shift, popt)
                    print(mse_n)



                    ###################################
                    #           CONSTRAINT ON A
                    ###################################

                    print("CONSTRAINT A\n")

                    # CONSTRAINT ON A
                    popt, pcov = curve_fit(func_norm_a, charge_dac_bins2, tot_mean_shift,
                        p0 = [2, 100, -10],bounds=([-100,0,-40], [100,1000, 80]),
                        maxfev=10000)
                    perr = np.sqrt(np.diag(pcov))

                    print(*popt)
                    print(*perr)

                    plt.pcolormesh(
                        occupancy_shift, np.linspace(-0.5, 127.5, 129, endpoint=True),
                        hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

                    aa = (popt[1])/(th_140*(th_140-popt[2])) - (popt[0])/(th_140)
                    # daa =
                    #a = (c)/(th_140*(th_140-t)) - (b)/(th_140)

                    plt.plot(y, func_norm_a(y, *popt), "r-", label=f"fit:\na ={aa}\n"
                        f"b = {ufloat(round(popt[0],3),round(perr[0],3))}\nc = {ufloat(np.around(popt[1],3),round(perr[1], 3))}\nt = {ufloat(np.around(popt[2],3),round(perr[2], 3))}")
                    plt.xlim([0, 250])
                    plt.ylim([0, 60])

                    plt.title("Constraint on parameter a")
                    plt.suptitle(f"ToT curve fit ({name})")
                    plt.xlabel("True injected charge [DAC]")
                    plt.ylabel("ToT [25 ns]")
                    set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                    cb = integer_ticks_colorbar()
                    cb.set_label("Hits / bin")
                    plt.legend(loc="upper left")
                    plt.savefig(f"Tot_fit_{name}_a(200).png")
                    pdf.savefig(); plt.clf()

                    print(popt)
                    mse_a = mse(func_norm_a, charge_dac_bins2, tot_mean_shift, popt)
                    print(mse_a)



                    ###################################
                    #           CONSTRAINT ON B
                    ###################################

                    print("CONSTRAIN B\n")

                    #p0 = [0.15, 2, 100, -10],bounds=([0 , -100, 0, -40], [0.3, 100,1000, 80]),

                    # CONSTRAINT ON B
                    popt, pcov = curve_fit(func_norm_b, charge_dac_bins2, tot_mean_shift,
                        p0 = [0.15, 100, -10],bounds=([0,0,-40], [0.3,1000, 80]),
                        maxfev=10000)
                    perr = np.sqrt(np.diag(pcov))

                    print(*popt)
                    print(*perr)

                    plt.pcolormesh(
                        occupancy_shift, np.linspace(-0.5, 127.5, 129, endpoint=True),
                        hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF


                    # (c)/(th_140-t) - a*th_140
                    bb = (popt[1])/(th_140-popt[2]) - popt[0]*th_140
                    # dbb =


                    plt.plot(y, func_norm_b(y, *popt), "r-", label=f"fit:\na ={ufloat(round(popt[0], 3), round(perr[0], 3))}\n"
                        f"b = {bb}\nc = {ufloat(np.around(popt[1],3),round(perr[1], 3))}\nt = {ufloat(np.around(popt[2],3),round(perr[2], 3))}")
                    plt.xlim([0, 250])
                    plt.ylim([0, 60])

                    plt.title("Constraint on parameter b")
                    plt.suptitle(f"ToT curve fit ({name})")
                    plt.xlabel("True injected charge [DAC]")
                    plt.ylabel("ToT [25 ns]")
                    set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                    cb = integer_ticks_colorbar()
                    cb.set_label("Hits / bin")
                    plt.legend(loc="upper left")
                    plt.savefig(f"Tot_fit_{name}_b(200).png")
                    pdf.savefig(); plt.clf()

                    mse_b = mse(func_norm_b, charge_dac_bins2, tot_mean_shift, popt)
                    print(mse_b)


                    ###################################
                    #           CONSTRAINT ON C
                    ###################################

                    print("CONSTRAINT C\n")

                    # CONSTRAINT ON C
                    popt, pcov = curve_fit(func_norm_c, charge_dac_bins2, tot_mean_shift,
                        p0 = [0.15, 2, -10],bounds=([0 , -100, -40], [0.3, 100, 80]),
                        maxfev=10000)
                    perr = np.sqrt(np.diag(pcov))

                    print(*popt)
                    print(*perr)

                    plt.pcolormesh(
                        occupancy_shift, np.linspace(-0.5, 127.5, 129, endpoint=True),
                        hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

                    #(th_140-t)*(a*th_140 + b)
                    cc = (th_140-popt[2])*(popt[0]*th_140 + popt[1])
                    #dcc =

                    plt.plot(y, func_norm_c(y, *popt), "r-", label=f"fit:\na ={ufloat(round(popt[0], 3), round(perr[0], 3))}\n"
                        f"b = {ufloat(round(popt[1],3),round(perr[1],3))}\nc = {cc}\nt = {ufloat(np.around(popt[2],3),round(perr[2], 3))}")
                    plt.xlim([0, 250])
                    plt.ylim([0, 60])

                    plt.title("Constraint on parameter c")
                    plt.suptitle(f"ToT curve fit ({name})")
                    plt.xlabel("True injected charge [DAC]")
                    plt.ylabel("ToT [25 ns]")
                    set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                    cb = integer_ticks_colorbar()
                    cb.set_label("Hits / bin")
                    plt.legend(loc="upper left")
                    plt.savefig(f"Tot_fit_{name}_c(200).png")
                    pdf.savefig(); plt.clf()

                    mse_c = mse(func_norm_c, charge_dac_bins2, tot_mean_shift, popt)
                    print(mse_c)



                    ###################################
                    #           CONSTRAINT ON T
                    ###################################

                    print("CONSTRAINT T\n")

                    # p0 = [0.15, 2, 100, -10],bounds=([0 , -100, 0, -40], [0.3, 100,1000, 80]),

                    # CONSTRAINT ON T
                    popt, pcov = curve_fit(func_norm_t, charge_dac_bins2, tot_mean_shift,
                        p0 = [0.15, 2, 100],bounds=([0 , -100, 0], [0.3, 100, 1000]),
                        maxfev=10000)
                    perr = np.sqrt(np.diag(pcov))

                    print(*popt)
                    print(*perr)

                    plt.pcolormesh(
                        occupancy_shift, np.linspace(-0.5, 127.5, 129, endpoint=True),
                        hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

                    #  th_140 - (c)/(a*th_140 + b)
                    tt = th_140 - (popt[2])/(popt[0]*th_140 + popt[1])
                    #dtt =

                    plt.plot(y, func_norm_t(y, *popt), "r-", label=f"fit:\na ={ufloat(round(popt[0], 3), round(perr[0], 3))}\n"
                        f"b = {ufloat(round(popt[1],3),round(perr[1],3))}\nc = {ufloat(np.around(popt[2],3),round(perr[2], 3))}\nt = {tt}")
                    plt.xlim([0, 250])
                    plt.ylim([0, 60])

                    plt.title("Constraint on parameter t")
                    plt.suptitle(f"ToT curve fit ({name})")
                    plt.xlabel("True injected charge [DAC]")
                    plt.ylabel("ToT [25 ns]")
                    set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                    cb = integer_ticks_colorbar()
                    cb.set_label("Hits / bin")
                    plt.legend(loc="upper left")
                    plt.savefig(f"Tot_fit_{name}_t(200).png")
                    pdf.savefig(); plt.clf()

                    mse_t = mse(func_norm_t, charge_dac_bins2, tot_mean_shift, popt)
                    print(mse_t)

                    # popt, pcov = curve_fit(func_casc2, charge_dac_bins2, tot_mean_shift,
                    #     p0 = [0.15, 2, -10],bounds=([0 , -100, -40], [0.3, 100, 80]),
                    #     maxfev=10000)
                    # perr = np.sqrt(np.diag(pcov))

                    # print(*popt)
                    # print(*perr)
                    #
                    # plt.pcolormesh(
                    #     occupancy_shift, np.linspace(-0.5, 127.5, 129, endpoint=True),
                    #     hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

                    # y = np.arange(mean[i-1]-29.01, 250, 1)
                    # plt.plot(y, func_casc2(y, *popt), "r-", label=f"fit {name}:\na ={ufloat(round(popt[0], 3), round(perr[0], 3))}\nb = {ufloat(round(popt[1],3),round(perr[1],3))}\nt = {ufloat(np.around(popt[2],3),round(perr[2], 3))}")
                    # plt.xlim([0, 250])
                    # plt.ylim([0, 60])
                    #
                    # plt.title(subtitle)
                    # plt.suptitle(f"ToT curve fit ({name})")
                    # plt.xlabel("True injected charge [DAC]")
                    # plt.ylabel("ToT [25 ns]")
                    # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                    # cb = integer_ticks_colorbar()
                    # cb.set_label("Hits / bin")
                    # plt.legend(loc="upper left")
                    # pdf.savefig();
                    # plt.savefig(f"tot_fit_{name}({the_vh}).png"); plt.clf()


            ################################### FIT #############
                # print(i, mean[i-1])
                # def func_casc(x,a,b,c,t):
                #     return np.where(x<mean[i-1], 0, np.maximum(0, a*x+b-(c/(x-t))))
                #
                #
                # mask_tot = np.isfinite(tot_mean)
                # tot_mean = tot_mean[mask_tot]
                # occu = occupancy_edges[2][:-1]
                # charge_dac_bins2 = occu[mask_tot]
                #
                #
                # if the_vh==140:
                #     popt, pcov = curve_fit(func_casc, charge_dac_bins2, tot_mean,
                #         p0 = [0.15, 2, 100, 1],bounds=([0 , -100, 0, 0], [0.2, 100,1000, 80]), maxfev=10000)
                #     #low = [0 , -100, 0, 0]
                #     #up = [0.2, 100,1000, 80]
                # elif the_vh==200:
                #     popt, pcov = curve_fit(func_casc, charge_dac_bins2, tot_mean,
                #         p0 = [0.15, 2, 100, 10],bounds=([0 , -100, 0, 0], [0.3, 100,10000, 80]), maxfev=10000)
                # else:
                #     print("Define new range.")
                # perr = np.sqrt(np.diag(pcov))
                #
                # print(*popt)
                # print(*perr)
                #
                # plt.pcolormesh(
                #     occupancy_edges[2], np.linspace(-0.5, 127.5, 129, endpoint=True),
                #     hist.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF
                #
                # y = np.arange(mean[i-1]-0.01, 250, 1)
                # plt.plot(y, func_casc(y, *popt), "r-", label=f"fit {name}:\na ={ufloat(round(popt[0], 3), round(perr[0], 3))}\nb = {ufloat(round(popt[1],3),round(perr[1],3))}\nc = {ufloat(round(popt[2],3),round(perr[2], 3))}\nt = {ufloat(np.around(popt[3],3),round(perr[3], 3))}")
                # plt.xlim([0, 250])
                # plt.ylim([0, 60])
                #
                # plt.title(subtitle)
                # plt.suptitle(f"ToT curve fit ({name})")
                # plt.xlabel("True injected charge [DAC]")
                # plt.ylabel("ToT [25 ns]")
                # set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
                # cb = integer_ticks_colorbar()
                # cb.set_label("Hits / bin")
                # plt.legend(loc="upper left")
                # pdf.savefig(); plt.clf()


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

        noise_DAC = np.sqrt(average((occupancy_charges - np.expand_dims(threshold_DAC, -1))**2, axis=2, weights=w, invalid=0))
        del w

        # Noise hist
        m = int(np.ceil(noise_DAC.max(initial=0, where=np.isfinite(noise_DAC)))) + 1
        for i, (fc, lc, name) in enumerate(FRONTENDS):
            if fc >= col_stop or lc < col_start:
                continue
            fc = max(0, fc - col_start)
            lc = min(col_n - 1, lc - col_start)
            ns = noise_DAC[fc:lc+1,:]
            noise_mean = ufloat(np.mean(ns[ns>0]), np.std(ns[ns>0], ddof=1))
            plt.hist(ns.reshape(-1), bins=min(20*m, 100), range=[0, m],
                     label=f"{name} ${noise_mean:L}$", histtype='step', color=f"C{i}")
        plt.title(subtitle)
        plt.suptitle(f"Noise (width of s-curve slope) distribution")
        plt.xlabel("Noise [DAC]")
        plt.ylabel("Pixels / bin")
        set_integer_ticks(plt.gca().yaxis)
        plt.grid(axis='y')
        plt.legend()
        pdf.savefig(); plt.clf()

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

    # Save data in npz file
    np.savez_compressed(
        "all_thresholds_HVs.npz",
        all_th = threshold_DAC,
        all_noise = noise_DAC,
        all_tot_HVC = tot_hist[3],
        all_tot_HV = tot_hist[4],
        all_occup = occupancy)
    print("\"*.npz\" file is created.")

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
    parser.add_argument("-totfit","--totfit", action="store_true")
    parser.add_argument("-thfit","--thfit", action="store_true")
    parser.add_argument("-thi","--thfile", help="The _threshold.npz file(s).")
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
