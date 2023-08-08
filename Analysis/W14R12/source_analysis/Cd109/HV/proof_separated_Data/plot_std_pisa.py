#!/usr/bin/env python3
"""Standard plots like hitmap and ToT histogram (HistOcc and HistToT not required)."""
import argparse
import glob
import os
import traceback
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from tqdm import tqdm
from plot_utils_pisa import *


# FIRST STEP: create a npz file of all data files (*_interpreted.h5)

#python3 plot_std_pisa.py -j fe55_allHV.pdf 20221011_110840_source_scan_interpreted.h5
#20221011_113405_source_scan_interpreted.h5 20221011_122440_source_scan_interpreted.h5
#20221011_124109_source_scan_interpreted.h5 20221011_125911_source_scan_interpreted.h5
#20221011_132146_source_scan_interpreted.h5 -f

#SECOND STEP: TRY npz file

#python3 plot_std_pisa.py -j fe55_allHV.pdf 20221011_110840_source_scan_interpreted.h5
#20221011_113405_source_scan_interpreted.h5 20221011_122440_source_scan_interpreted.h5
#20221011_124109_source_scan_interpreted.h5 20221011_125911_source_scan_interpreted.h5
#20221011_132146_source_scan_interpreted.h5 -f -npz tot_fe_HVfiles.npz



def is_single_hit_event(timestamps, window_us=3.2):
    """Returns a mask that selects hits from single-hit events only.

    An event here is defined as a series of hits happening within a
    window of window_us microseconds. The timestamp is used to
    determine the arrival time of the hit, which is very approximate
    and only works if the hit rate is low (i.e. no noisy pixels!).

    Example usage:
        f = tb.open_file("..._interpreted.h5")
        hits = f.root.Dut[:]
        mask = is_single_hit_event(hits["timestamp"])
        single_hits = hits[mask]
    """
    window = int(window_us * 40)
    diff = np.diff(np.concatenate(((-2**63,), timestamps, (2**63-1,))))
    diff_from_previous = diff[:-1]
    diff_from_next = diff[1:]
    min_diff = np.minimum(diff_from_next, diff_from_previous)
    return min_diff > window


def main(input_files, overwrite=False, log_tot=False, output_file=None):
    if output_file is None:
        output_file = os.path.splitext(input_files[0])[0] + ".pdf"
    if os.path.isfile(output_file) and not overwrite:
        return

    if args.npz is None:
        print("We are in npz=None")
        # Prepare histograms
        counts2d = np.zeros((512, 512))
        tot1d = [np.zeros(128) for _ in range(len(FRONTENDS))]
        tot1d_single_hits = [np.zeros(128) for _ in range(len(FRONTENDS))]
        tot2d = np.zeros((512, 512))
        counts2d16 = np.zeros((32, 32))
        counts = np.zeros((512, 512, 128))
        cfg = []
        n_total_hits = 0


        for input_file in tqdm(input_files, disable=len(input_files)<2):
            print("Processing", input_file)
            with tb.open_file(input_file) as f:

                try:
                    cfg.append(get_config_dict(f))
                except tb.NoSuchNodeError:
                    print("NoSuchNodeError")
                    pass

                try:
                    n_hits = f.root.Dut.shape[0]
                except tb.NoSuchNodeError:
                    print("NoSuchNodeError2")
                    continue
                n_total_hits += n_hits

                # n_hits = f.root.Dut.shape[0]
                # n_total_hits += n_hits

                all_zeros = not np.any(counts)
                print(all_zeros)

                # Process one chunk of data at a time
                csz = 2**24
                for i_first in tqdm(range(0, n_hits, csz), unit="chunk", disable=n_hits/csz<=1):
                    i_last = min(n_hits, i_first + csz)

                    # Load hits
                    hits = f.root.Dut[i_first:i_last]
                    with np.errstate(all='ignore'):
                        tot = (hits["te"] - hits["le"]) & 0x7f
                    fe_masks = [(hits["col"] >= fc) & (hits["col"] <= lc) for fc, lc, _ in FRONTENDS]
                    single_hits_mask = is_single_hit_event(hits["timestamp"])

                    counts2d_tmp, counts2d_edges, _ = np.histogram2d(
                        hits["col"], hits["row"], bins=[512, 512], range=[[0, 512], [0, 512]])
                    counts2d += counts2d_tmp
                    del counts2d_tmp

                    for i, mask in enumerate(fe_masks):
                        tot1d_tmp, tot1d_edges = np.histogram(
                            tot[mask], bins=128, range=[-0.5, 127.5])
                        tot1d[i] += tot1d_tmp
                        tot1d_tmp, tot1d_edges = np.histogram(
                            tot[single_hits_mask & mask], bins=128, range=[-0.5, 127.5])
                        tot1d_single_hits[i] += tot1d_tmp
                        del tot1d_tmp

                    tot2d_tmp, tot2d_edges, _  = np.histogram2d(
                        hits["col"], hits["row"], bins=[512, 512], range=[[0, 512], [0, 512]],
                        weights=tot)
                    tot2d += tot2d_tmp
                    del tot2d_tmp

                    counts2d16_tmp, edges16, _ = np.histogram2d(
                        hits["col"], hits["row"], bins=[32, 32], range=[[0, 512], [0, 512]])
                    counts2d16 += counts2d16_tmp
                    del counts2d16_tmp




                    # For peak.py
                    with np.errstate(all='ignore'):
                        tmp, edges = np.histogramdd(
                            (hits["col"], hits["row"], (hits["te"] - hits["le"]) & 0x7f),
                            bins=[512, 512, 128], range=[[0, 512], [0, 512], [0, 128]])
                        counts += tmp


                    del hits, tot, fe_masks, tmp

            all_zeros = not np.any(counts)
            print(all_zeros)
            print(n_total_hits)

        # Create npz of all inputs files

        np.savez_compressed(
            "tot_cd_HVfiles.npz",
            counts = counts2d,
            counts2 = counts2d16,
            counts2e = counts2d_edges,
            tot = tot1d,
            tot_single_hits = tot1d_single_hits,
            tot1_edges = tot1d_edges,
            tot2 = tot2d,
            tot2_edges = tot2d_edges,
            edges = edges16,
            conf=cfg,
            nhits = n_total_hits,
            counts_peak = counts)
        print("\"*.npz\" file is created.")
    else:
        counts2d = np.zeros((512, 512))
        counts2d16 = np.zeros((32, 32))
        tot1d = [np.zeros(128) for _ in range(len(FRONTENDS))]
        tot1d_single_hits = [np.zeros(128) for _ in range(len(FRONTENDS))]
        tot2d = np.zeros((512, 512))

        with np.load(args.npz, allow_pickle=True) as data:
            counts2d = data['counts']
            counts2d16 = data['counts2']
            counts2d_edges = data['counts2e']
            tot1d = data['tot']
            tot1d_single_hits = data['tot_single_hits']
            tot2d = data['tot2']
            tot1d_edges = data['tot1_edges']
            tot2d_edges = data['tot2_edges']
            edges16 = data['edges']
            cfg = data['conf']
            n_total_hits = data['nhits']




    with PdfPages(output_file) as pdf:
        plt.figure(figsize=(6.4, 4.8))

        if len(input_files) > 1:
            plt.annotate(
                split_long_text(
                    "This file was generated by joining the following\n\n"
                    + "\n".join(input_files)
                    ), (0.5, 0.5), ha='center', va='center')
            plt.gca().set_axis_off()
            pdf.savefig(); plt.clf()

        for input_file, c in zip(input_files, cfg):
            draw_summary(input_file, c)
            pdf.savefig(); plt.clf()
        # print("Summary")

        if n_total_hits == 0:
            plt.annotate("No hits recorded!", (0.5, 0.5), ha='center', va='center')
            plt.gca().set_axis_off()
            pdf.savefig(); plt.clf()
            return

        # Histogram of hits per pixel
        m = np.quantile(counts2d[counts2d > 0], 0.99) * 1.2 if np.any(counts2d > 0) else 1
        bins = 100 if m > 200 else int(max(m, 5))
        for fc, lc, name in FRONTENDS:
            plt.hist(counts2d[fc:lc+1,:].reshape(-1), label=name, histtype='step',
                     bins=bins, range=[0.5, max(m, 5) + 0.5])
        plt.title("Hits per pixel")
        plt.xlabel("Number of hits")
        plt.ylabel("Pixels / bin")
        plt.grid(axis='y')
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        plt.legend()
        pdf.savefig(); plt.clf()
        # print("Hits hist")

        # Histogram of ToT
        for (_, _, name), hist in zip(FRONTENDS, tot1d):
            plt.step((tot1d_edges[1:] + tot1d_edges[:-1]) / 2,
                     hist, where='mid', label=name)
        plt.title("ToT")
        plt.xlabel("ToT [25 ns]")
        plt.xlim([10,128])
        plt.ylim([0,100])
        plt.ylabel("Hits / bin")
        plt.grid(axis='y')
        if log_tot:
            plt.yscale('log')
            set_integer_ticks(plt.gca().xaxis)
        else:
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        plt.legend()
        pdf.savefig(); plt.clf()
        # print("ToT Hist")

        # Histogram of ToT (single hits)
        for (_, _, name), hist in zip(FRONTENDS, tot1d_single_hits):
            plt.step((tot1d_edges[1:] + tot1d_edges[:-1]) / 2,
                     hist, where='mid', label=name)
        plt.title("ToT (isolated hits only)")
        plt.xlabel("ToT [25 ns]")
        plt.ylabel("Hits / bin")
        plt.grid(axis='y')
        if log_tot:
            plt.yscale('log')
            set_integer_ticks(plt.gca().xaxis)
        else:
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        plt.legend()
        pdf.savefig(); plt.clf()
        # print("ToT Hist")

        # Hit map
        plt.pcolormesh(counts2d_edges, counts2d_edges, counts2d.transpose(),
                       vmin=0, vmax=m, rasterized=True)  # Necessary for quick save and view in PDF
        plt.title("Hit map")
        plt.xlabel("Col")
        plt.ylabel("Row")
        cb = integer_ticks_colorbar()
        cb.set_label("Hits / pixel")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        frontend_names_on_top()
        pdf.savefig(); plt.clf()
        # print("Hitmap")

        # Map of the average ToT
        with np.errstate(all='ignore'):
            totavg = tot2d / counts2d
        plt.pcolormesh(tot2d_edges, tot2d_edges, totavg.transpose(),
                        vmin=-0.5, vmax=127.5, rasterized=True)  # Necessary for quick save and view in PDF
        plt.title("Average ToT map")
        plt.xlabel("Col")
        plt.ylabel("Row")
        cb = integer_ticks_colorbar()
        cb.set_label("ToT [25 ns]")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        frontend_names_on_top()
        pdf.savefig(); plt.clf()
        # print("ToT map")

        # Noisy pixels
        if all(c.get("configuration_in.scan.run_config.scan_id") == "source_scan" for c in cfg):
            MAX_RATE = 1  # Above this rate [Hz] pixels are marked noisy
            scan_time = 0
            for i, c in enumerate(cfg):
                try:
                    scan_time += float(c["configuration_in.scan.scan_config.scan_time"])
                except Exception:
                    print(f"WARNING: could not determine scan time from {input_files[i]}")
            max_hits = scan_time * MAX_RATE
            mask = counts2d > max_hits
            plt.axes((0.125, 0.11, 0.775, 0.72))
            plt.pcolormesh(counts2d_edges, counts2d_edges, 1 * mask.transpose(),
                           vmin=0, vmax=1, rasterized=True)  # Necessary for quick save and view in PDF
            plt.suptitle("Noisy pixels in yellow (ignore this plot if source was used)")
            plt.title(f"Noisy means rate > {MAX_RATE:.3g} Hz")
            plt.xlabel("Col")
            plt.ylabel("Row")
            set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
            frontend_names_on_top()
            pdf.savefig(); plt.clf()

            # if np.count_nonzero(mask):
            #     noisy_list = np.argwhere(mask)
            #     noisy_indices = np.nonzero(mask)
            #     srt = np.argsort(-counts2d[noisy_indices])
            #     noisy_indices = tuple(x[srt] for x in noisy_indices)
            #     noisy_list = noisy_list[srt]
            #     mi = min(len(noisy_list), 100)
            #     tmp = "\n".join(
            #         ",    ".join(f"({a}, {b}) = {float(c)/scan_time:.3g}" for (a, b), c in g)
            #         for g in groupwise(zip(noisy_list[:mi], counts2d[tuple(x[:mi] for x in noisy_indices)]), 4))
            #     plt.annotate(
            #         split_long_text(
            #             "Noisiest pixels (col, row) = rate [Hz]\n"
            #             f"{tmp}"
            #             f'{", ..." if len(noisy_list) > mi else ""}'
            #             f"\nTotal = {len(noisy_list)} pixels"
            #         ), (0.5, 0.5), ha='center', va='center')
            # else:
            #     plt.annotate("No noisy pixel found.", (0.5, 0.5), ha='center', va='center')
            # plt.gca().set_axis_off()
            # pdf.savefig(); plt.clf()
        # print("Noisy")

        # Source positioning
        m = np.quantile(counts2d16[counts2d16 > 0], 0.99) * 1.2 if np.any(counts2d > 0) else 1
        cmap = matplotlib.cm.get_cmap("viridis").copy()
        cmap.set_over("r")
        plt.pcolormesh(edges16, edges16, counts2d16.transpose(), vmin=0, vmax=m,
                       cmap=cmap, rasterized=True)  # Necessary for quick save and view in PDF
        plt.title("Hit map in 16x16 regions for source positioning")
        plt.xlabel("Col")
        plt.ylabel("Row")
        cb = plt.colorbar()
        cb.set_label("Avg. hits / 16x16 region (red = out of scale)")
        set_integer_ticks(plt.gca().xaxis, plt.gca().yaxis)
        frontend_names_on_top()
        pdf.savefig(); plt.clf()
        # print("Source")

        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file", nargs="*",
        help="The _interpreted.h5 file(s). If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    parser.add_argument("-j", "--join", metavar="OUTPUT_FILE.PDF", default=None,
                        help="Join all input files and put results in the given PDF.")
    parser.add_argument("--log-tot", action="store_true",
                        help="Use log scale for ToT.")
    parser.add_argument("-npz", help="Use npz file.")
    args = parser.parse_args()
    print(args.join, args.npz)

    files = []
    if args.input_file:  # If anything was given on the command line
        for pattern in args.input_file:
            files.extend(glob.glob(pattern, recursive=True))
    else:
        files.extend(glob.glob("output_data/module_0/chip_0/*_interpreted.h5"))
    files.sort()
    print(files)

    if args.npz!=None:
        args.join = args.join.replace(".pdf", "_from_npz.pdf")

    print(args.join)

    if args.join is None:
        for fp in tqdm(files):
            try:
                main([fp], args.overwrite, args.log_tot)
            except Exception:
                print(traceback.format_exc())
    else:
        main(files, args.overwrite, args.log_tot, args.join)
