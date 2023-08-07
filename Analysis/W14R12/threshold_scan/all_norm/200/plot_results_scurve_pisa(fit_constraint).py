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

VIRIDIS_WHITE_UNDER = matplotlib.cm.get_cmap('viridis').copy()
VIRIDIS_WHITE_UNDER.set_under('w')


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
                        help="The _threshold_scan_interpreted_scurve.npz file(s). If \"all\" it takes automatically all npz files in the directory.")
    # parser.add_argument("output_file", help="The output PDF.")
    # parser.add_argument("input_file", nargs="+",
    #                     help="The _threshold_scan_interpreted_scurve.npz file(s).")
    args = parser.parse_args()

    files = []
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
            #print(col)

            tot+= data["tot"]
            #print(tot)

            overwritten = (~np.isnan(occupancy_npz)) & (~np.isnan(data['occup']))
            n_overwritten = np.count_nonzero(overwritten)
            if n_overwritten:
                print("WARNING Multiple values of threshold for the same pixel(s)")
                print(f"    count={n_overwritten}, file={fp}")
            occupancy_npz = np.where(np.isnan(occupancy_npz), data['occup'], occupancy_npz)

    np.savez_compressed(
        "all_thresholds_norm.npz",
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
            # if fc >= col_stop or lc < col_start:
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

        ##############################
        ###########         MIO
        ##############################


        for i, (fc, lc, name) in enumerate(FRONTENDS):
            # if fc >= col_stop[i] or lc < col_start[i]:
            #     continue
            th = thresholds[fc:lc+1,:]
            #print(th)
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
                # print(bin_center)
                # print(bin_height)
                popt, pcov = curve_fit(gauss, bin_center, bin_height, p0 = [0.2*entries, th_m, 3], bounds=(low, up))
                print(*popt)
                mean_b[i] = popt[1]
                perr = np.sqrt(np.diag(pcov))
                print(*perr)
                xb = np.arange(bin_edge[0], bin_edge[-1], 0.005)
                plt.plot(xb, gauss(xb, *popt), "r-", label=f"fit {name}:\nmean={ufloat(round(popt[1], 3), round(perr[1],3))}\nsigma={ufloat(round(popt[2], 3), round(perr[2],3))}")
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

                plt.xlim([50,110])
                pdf.savefig()
                plt.savefig(f"all_norm_thdist_200.png")
                plt.clf()


        plt.clf()
        # plt.suptitle("Threshold distribution")
        # plt.xlabel("Threshold [DAC]")
        # plt.ylabel("Pixels / bin")
        # set_integer_ticks(plt.gca().yaxis)
        # plt.legend(loc="upper left", fontsize=9)
        # plt.grid(axis='y')

            #Fit
        #     low = [0.1*entries,10, 0]
        #     up = [0.3*entries, 200, 30]
        #
        #     bin_center = (bin_edge[:-1]+bin_edge[1:])/2
        #     popt, pcov = curve_fit(gauss, bin_center, bin_height, p0 = [0.2*entries, th_m, 3], bounds=(low, up))
        #     print(*popt)
        #     mean[i] = popt[1]
        #     perr = np.sqrt(np.diag(pcov))
        #     print(*perr)
        #     xb = np.arange(bin_edge[0], bin_edge[-1], 0.005)
        #     plt.plot(xb, gauss(xb, *popt), "r-", label=f"fit {name}:\nmean={ufloat(round(popt[1], 3), round(perr[1],3))}\nsigma={ufloat(round(popt[2], 3), round(perr[2],3))}")
        #     #Save results in a txt file
        #     with open(f"th_fitresults[all {name}].txt", "w") as outf:
        #         print("#A#mean#sigma:", file=outf)
        #         print(*popt, file=outf)
        #         print("#SA#Smean#Ssigma:", file=outf)
        #         print(*perr, file=outf)
        #
        #
        # plt.suptitle("Threshold distribution")
        # plt.xlabel("Threshold [DAC]")
        # plt.ylabel("Pixels / bin")
        # set_integer_ticks(plt.gca().yaxis)
        # plt.legend(loc="upper left", fontsize=9)
        # plt.grid(axis='y')
        # pdf.savefig(); plt.clf()




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


        # for i, (fc, lc, name) in enumerate(FRONTENDS):
            #for j in range(0,8,1):
        plt.pcolormesh(
            charge_dac_edges[:-1], np.linspace(-0.5, 127.5, 128, endpoint=True),
            tot.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF
        plt.suptitle(f"ToT curve (Normal)")
        plt.xlabel("Injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = integer_ticks_colorbar()
        cb.set_label("Hits / bin")
        pdf.savefig(); plt.clf()

        ####################### TOT MEAN VS INJ CHARGE ##########

        tot_temp = np.tile(np.linspace(0, 127, 128, endpoint=True), (198,1))
        tot_mean= np.sum(tot_temp*tot,axis=1)/ np.sum(tot, axis=1)
        del tot_temp

        plt.plot(charge_dac_edges[:-1], tot_mean, rasterized=True)
        plt.suptitle(f"ToT curve (Normal)")
        plt.xlabel("Injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        plt.ylim([0,128])
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        pdf.savefig(); plt.clf()

        ########################################TOT SHIFT AND FIT ##############
        charge_shift = charge_dac_edges[:-1] - 29
        plt.pcolormesh(
            charge_shift, np.linspace(-0.5, 127.5, 128, endpoint=True),
            tot.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

        plt.suptitle(f"ToT curve (Normal)")
        plt.xlabel("True Injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = integer_ticks_colorbar()
        cb.set_label("Hits / bin")
        plt.xlim([0,200])
        pdf.savefig(); plt.clf()

        th_140 = 53.62
        #print(mean_b[0])
        #((th_140)**2)*a-(a*t*th_140)+(b*th_140)-(t*b)

        #print(mean_b)

        def mse(func, x, y, coefs):
            return np.mean((func(x, *coefs) - y)**2)



        ######################################
        #           NO CONSTRAINT
        ######################################
        def func_norm(x,a,b,c,t):
            return np.where(x<mean_b[0]-29, 0, np.maximum(0, a*x+b-(c/(x-t))))

        ######################################
        #           CONSTRAINT ON C
        ######################################
        # (th_140-t)*(a*th_140 + b)
        def func_norm_c2(x,a,b,t):
            return np.where(x<mean_b[0]-29, 0, np.maximum(0, a*x+b-((((th_140)**2)*a-(a*t*th_140)+(b*th_140)-(t*b))/(x-t))))

        def func_norm_c(x,a,b,t):
            return np.where(x<mean_b[0]-29, 0, np.maximum(0, a*x+b-(((th_140-t)*(a*th_140 + b))/(x-t))))


        ######################################
        #           CONSTRAINT ON A
        ######################################
        # (c)/(th_140*(th_140-t)) - (b)/(th_140)
        def func_norm_a(x,b,c,t):
            return np.where(x<mean_b[0]-29, 0, np.maximum(0, ((c)/(th_140*(th_140-t)) - (b)/(th_140))*x+b-(c/(x-t))))


        ######################################
        #           CONSTRAINT ON B
        ######################################
        # (c)/(th_140-t) - a*th_140
        def func_norm_b(x,a,c,t):
            return np.where(x<mean_b[0]-29, 0, np.maximum(0, a*x+((c)/(th_140-t) - a*th_140)-(c/(x-t))))


        ######################################
        #           CONSTRAINT ON T
        ######################################
        # th_140 - (c)/(a*th_140 + b)
        def func_norm_t(x,a,b,c):
            return np.where(x<mean_b[0]-29, 0, np.maximum(0, a*x+b-(c/(x-(th_140 - (c)/(a*th_140 + b))))))



        tot_mean_shift = tot_mean
        mask_tot = np.isfinite(tot_mean_shift)
        tot_mean_shift = tot_mean_shift[mask_tot]
        print(type(tot_mean_shift))
        occu = charge_shift
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
            charge_shift, np.linspace(-0.5, 127.5, 128, endpoint=True),
            tot.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

        print(*popt)
        print(*perr)

        y = np.arange(mean_b[0]-29.01, 189, 1)
        plt.plot(y, func_norm(y, *popt), "r-", label=f"fit:\na ={ufloat(round(popt[0], 3), round(perr[0], 3))}\n"
            f"b = {ufloat(round(popt[1],3),round(perr[1],3))}\nc = {ufloat(np.around(popt[2],3),round(perr[2], 3))}\nt = {ufloat(np.around(popt[3],3),round(perr[3], 3))}")

        plt.xlim([0, 250])
        plt.ylim([0, 60])
        plt.suptitle(f"ToT curve (Normal)")
        plt.xlabel("True injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = integer_ticks_colorbar()
        cb.set_label("Hits / bin")
        plt.legend(loc="upper left")
        pdf.savefig();
        plt.savefig("Tot_fit_normal(200).png"); plt.clf()

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
            charge_shift, np.linspace(-0.5, 127.5, 128, endpoint=True),
            tot.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

        aa = (popt[1])/(th_140*(th_140-popt[2])) - (popt[0])/(th_140)
        # daa =
        #a = (c)/(th_140*(th_140-t)) - (b)/(th_140)

        plt.plot(y, func_norm_a(y, *popt), "r-", label=f"fit:\na ={aa}\n"
            f"b = {ufloat(round(popt[0],3),round(perr[0],3))}\nc = {ufloat(np.around(popt[1],3),round(perr[1], 3))}\nt = {ufloat(np.around(popt[2],3),round(perr[2], 3))}")
        plt.xlim([0, 250])
        plt.ylim([0, 60])

        plt.title("Constraint on parameter a")
        plt.suptitle(f"ToT curve fit (Normal)")
        plt.xlabel("True injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = integer_ticks_colorbar()
        cb.set_label("Hits / bin")
        plt.legend(loc="upper left")
        plt.savefig("Tot_fit_normal_a(200).png")
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
            charge_shift, np.linspace(-0.5, 127.5, 128, endpoint=True),
            tot.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF


        # (c)/(th_140-t) - a*th_140
        bb = (popt[1])/(th_140-popt[2]) - popt[0]*th_140
        # dbb =


        plt.plot(y, func_norm_b(y, *popt), "r-", label=f"fit:\na ={ufloat(round(popt[0], 3), round(perr[0], 3))}\n"
            f"b = {bb}\nc = {ufloat(np.around(popt[1],3),round(perr[1], 3))}\nt = {ufloat(np.around(popt[2],3),round(perr[2], 3))}")
        plt.xlim([0, 250])
        plt.ylim([0, 60])

        plt.title("Constraint on parameter b")
        plt.suptitle(f"ToT curve fit (Normal)")
        plt.xlabel("True injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = integer_ticks_colorbar()
        cb.set_label("Hits / bin")
        plt.legend(loc="upper left")
        plt.savefig("Tot_fit_normal_b(200).png")
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
            charge_shift, np.linspace(-0.5, 127.5, 128, endpoint=True),
            tot.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

        #(th_140-t)*(a*th_140 + b)
        cc = (th_140-popt[2])*(popt[0]*th_140 + popt[1])
        #dcc =

        plt.plot(y, func_norm_c(y, *popt), "r-", label=f"fit:\na ={ufloat(round(popt[0], 3), round(perr[0], 3))}\n"
            f"b = {ufloat(round(popt[1],3),round(perr[1],3))}\nc = {cc}\nt = {ufloat(np.around(popt[2],3),round(perr[2], 3))}")
        plt.xlim([0, 250])
        plt.ylim([0, 60])

        plt.title("Constraint on parameter c")
        plt.suptitle(f"ToT curve fit (Normal)")
        plt.xlabel("True injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = integer_ticks_colorbar()
        cb.set_label("Hits / bin")
        plt.legend(loc="upper left")
        plt.savefig("Tot_fit_normal_c(200).png")
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
            charge_shift, np.linspace(-0.5, 127.5, 128, endpoint=True),
            tot.transpose(), vmin=1, cmap=VIRIDIS_WHITE_UNDER, rasterized=True)  # Necessary for quick save and view in PDF

        #  th_140 - (c)/(a*th_140 + b)
        tt = th_140 - (popt[2])/(popt[0]*th_140 + popt[1])
        #dtt =

        plt.plot(y, func_norm_t(y, *popt), "r-", label=f"fit:\na ={ufloat(round(popt[0], 3), round(perr[0], 3))}\n"
            f"b = {ufloat(round(popt[1],3),round(perr[1],3))}\nc = {ufloat(np.around(popt[2],3),round(perr[2], 3))}\nt = {tt}")
        plt.xlim([0, 250])
        plt.ylim([0, 60])

        plt.title("Constraint on parameter t")
        plt.suptitle(f"ToT curve fit (Normal)")
        plt.xlabel("True injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        cb = integer_ticks_colorbar()
        cb.set_label("Hits / bin")
        plt.legend(loc="upper left")
        plt.savefig("Tot_fit_normal_t(200).png")
        pdf.savefig(); plt.clf()

        mse_t = mse(func_norm_t, charge_dac_bins2, tot_mean_shift, popt)
        print(mse_t)



        # S-Curve as 2D histogram
        occupancy_charges = charge_dac_edges.astype(np.float32)
        occupancy_charges = (occupancy_charges[:-1] + occupancy_charges[1:]) / 2
        occupancy_charges = np.tile(occupancy_charges, (512, 512, 1))
        charge_dac_range = [min(charge_dac_edges) - 0.5, max(charge_dac_edges) + 0.5 -1]

        for i, (fc, lc, name) in enumerate(FRONTENDS):
        #     # if fc >= col_stop or lc < col_start:
        #     #     continue
        #     # fc = max(0, fc - col_start)
        #     # lc = min(col_stop-col_start - 1, lc - col_start)
            if name=="Normal":
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
                plt.savefig("all_norm_thscan_200.png")
                pdf.savefig(); plt.clf()
