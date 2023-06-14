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
from uncertainties import ufloat

def average(a, axis=None, weights=1, invalid=np.NaN):
    """Like np.average, but returns `invalid` instead of crashing if the sum of weights is zero."""
    return np.nan_to_num(np.sum(a * weights, axis=axis).astype(float) / np.sum(weights, axis=axis).astype(float), nan=invalid)


def main(infile, overwrite=False, frontend="cascode"):
    outfile = os.path.splitext(infile)[0] + f"_totfit(cut_{args.cut}).pdf"
    #if os.path.isfile(output_file) and not overwrite:
    #    return
    logging.info(f"Starting the analysis of {infile}")

    # Loading information on the chip configuration
    with tb.open_file(infile) as f, PdfPages(outfile) as pdf:
        cfg = get_config_dict(f)
        chip_serial_number = cfg["configuration_in.chip.settings.chip_sn"]
        plt.figure(figsize=(6.4,4.8))
        s=""
        param = ["IBIAS", "ITHR", "ICASN", "IDB", "ITUNE", "VRESET", "VCASC", "VCASP", "VCLIP"]
        for i in param:
            value = cfg[f"configuration_in.chip.registers.{i}"]
            s+=f"{i} = {value}, "
        plt.annotate(
            split_long_text(f"{os.path.abspath(infile)}\n"
                            f"Chip {chip_serial_number}\n"
                            f"Version {get_commit()}\n"
                            f"Registers: {s[:-2]}"),
            (0.5, 0.5), ha='center', va='center')
        plt.gca().set_axis_off()
        pdf.savefig(); plt.clf()
        logging.debug("Chip information are retrieved.")

        a = s.find("VCLIP")
        vclip = s[a:a+11]

        # Load hits
        hits = f.root.Dut[:]
        #print(hits)
        with np.errstate(all="ignore"):
            tot = (hits["te"] - hits["le"]) & 0x7f          #BITWISE AND
            tot2 = np.where(tot<args.cut, tot, np.nan)

        logging.debug("Hits are saved and tot are calculated.")

        # Load information on injected charge and steps taken
        sp = f.root.configuration_in.scan.scan_params[:]
        scan_params = np.zeros(sp["scan_param_id"].max() +1, dtype=sp.dtype)
        for i in range(len(scan_params)):
            m = sp["scan_param_id"]== i
            if np.any(m):
                scan_params[i] = sp[m.argmax()]
            else:
                scan_params[i]["scan_param_id"] = i
        del sp

        vh = scan_params["vcal_high"][hits["scan_param_id"]]
        vl = scan_params["vcal_low"][hits["scan_param_id"]]
        del scan_params
        logging.debug("VH and VL and relative steps are saved.")

        charge_dac = vh - vl
        n_injections = int(cfg["configuration_in.scan.scan_config.n_injections"])
        the_vh = int(cfg["configuration_in.scan.scan_config.VCAL_HIGH"])
        start_vl = int(cfg["configuration_in.scan.scan_config.VCAL_LOW_start"])
        stop_vl = int(cfg["configuration_in.scan.scan_config.VCAL_LOW_stop"])
        step_vl = int(cfg["configuration_in.scan.scan_config.VCAL_LOW_step"])
        charge_dac_values = [
            the_vh - x for x in range(start_vl, stop_vl, step_vl)]

        # for i in range(0,len(charge_dac_values)):
        #      charge_dac_values[i] -=29

        subtitle = f"VH = {the_vh}, VL = {start_vl}..{stop_vl} (step {step_vl})"
        logging.debug("Charge values are saved.")

        charge_dac_bins = len(charge_dac_values)
        charge_dac_range = [min(charge_dac_values) - 0.5, max(charge_dac_values) + 0.5]
        # Count hits per pixel per injected charge value
        row_start = int(cfg["configuration_in.scan.scan_config.start_row"])
        row_stop = int(cfg["configuration_in.scan.scan_config.stop_row"])
        col_start = int(cfg["configuration_in.scan.scan_config.start_column"])
        col_stop = int(cfg["configuration_in.scan.scan_config.stop_column"])
        row_n, col_n = row_stop - row_start, col_stop - col_start
        occupancy, occupancy_edges = np.histogramdd(
            (hits["col"], hits["row"], charge_dac),
            bins=[col_n, row_n, charge_dac_bins],
            range=[[col_start, col_stop], [row_start, row_stop], charge_dac_range])
        occupancy /= n_injections


        occupancy_charges = occupancy_edges[2].astype(np.float32)
        occupancy_charges = (occupancy_charges[:-1] + occupancy_charges[1:]) / 2
        occupancy_charges = np.tile(occupancy_charges, (col_n, row_n, 1))

################################################################################
        # PLOT TOT
        #m = 32 if tot.max() <= 32 else 128
        ch_lim = the_vh-1
        m=32
        # Saving histo and data_bin in variables
        # histo, charge_bins, tot_bins, _ = plt.hist2d(charge_dac, tot, bins=[250, m],
        #     range=[[-0.5, 249.5], [-0.5, m - 0.5]],
        #     cmin=1, rasterized=True)
        histo, charge_bins, tot_bins, _ = plt.hist2d(charge_dac, tot, bins=[ch_lim, m],
            range=[[0, ch_lim], [0, m]], cmin=1, rasterized=True)
        np.nan_to_num(histo, copy=False)
        #print(histo)

        # Taking the central value of each bin (chearge and tot)
        charge_bins = (charge_bins[:-1] + charge_bins[1:])/2
        tot_bins = (tot_bins[:-1] + tot_bins[1:])/2

        # Weighted mean of tot for eah charge_dac
        # Creating an array with the repetion of all tot_bins for each value of charge_dac
        # Then it does the average on the second axis, so y.
        tot_mean_temp = np.tile(tot_bins, (len(charge_bins),1))
        with np.errstate(invalid='ignore', divide='ignore'):
            tot_mean = np.sum(tot_mean_temp*histo,axis=1)/ np.sum(histo, axis=1)

        # Removing point with no hits
        mask = np.isfinite(tot_mean)
        tot_mean = tot_mean[mask]
        charge_bins = charge_bins[mask]
        #tot_mean=tot_mean[tot_mean<18]

        plt.title(subtitle)
        plt.suptitle("ToT curve")
        plt.xlabel("Injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        cb = plt.colorbar()
        cb.set_label("Hits / bin")
        pdf.savefig(); plt.clf()



        if args.shift:
            # PLOT TOT SHIFTED
            # Shift in charge(injection circuitry saturation) obtained by fitting the
            # threshold distributions for VH=140 (no saturation) and VH=200.
            # DeltaQ = 29 is the difference between their mean values.
            charge_dac = vh - vl - 29
            histo, charge_bins, tot_bins, _ = plt.hist2d(charge_dac, tot, bins=[250, m],
                range=[[-0.5, 249.5], [-0.5, m - 0.5]],
                cmin=1, rasterized=True)
            np.nan_to_num(histo, copy=False)

            # Taking the central value of each bin (chearge and tot)
            charge_bins = (charge_bins[:-1] + charge_bins[1:])/2
            tot_bins = (tot_bins[:-1] + tot_bins[1:])/2

            # Weighted mean of tot for eah charge_dac
            # Creating an array with the repetion of all tot_bins for each value of charge_dac
            # Then it does the average on the second axis, so y.
            tot_mean_a = np.tile(tot_bins, (len(charge_bins),1))
            with np.errstate(invalid='ignore', divide='ignore'):
                tot_mean = np.sum(tot_mean_a*histo,axis=1)/ np.sum(histo, axis=1)

            #Removing point with no hits
            mask = np.isfinite(tot_mean)
            tot_mean = tot_mean[mask]
            charge_bins = charge_bins[mask]

            plt.title(subtitle)
            plt.suptitle("ToT curve")
            plt.xlabel("True injected charge [DAC]")
            plt.ylabel("ToT [25 ns]")
            cb = plt.colorbar()
            cb.set_label("Hits / bin")
            pdf.savefig(); plt.clf()


###############################################################################
#############    FIT THRESHOLD DISTRIBUTION    ################################

        # Compute the threshold for each pixel as the weighted average
        # of the injected charge, where the weights are given by the
        # occupancy such that occu = 0.5 has weight 1, occu = 0,1 have
        # weight 0, and anything in between is linearly interpolated
        # Assuming the shape is an erf, this estimator is consistent
        w = np.maximum(0, 0.5 - np.abs(occupancy - 0.5))
        threshold_DAC = average(occupancy_charges, axis=2, weights=w)
        m1 = int(max(charge_dac_range[0], threshold_DAC.min() - 2))
        m2 = int(min(charge_dac_range[1], threshold_DAC.max() + 2))
        bin_height, bin_edge, _ = plt.hist(threshold_DAC.reshape(-1), bins=m2-m1, range=[m1, m2])
        bin_center = (bin_edge[:-1]+bin_edge[1:])/2# -0.5

        if frontend=="normal" or frontend=="cascode":

        #Range for fit parameters
            low = [0,10, 0]
            up = [500, 200, 30]

        #Find the center value among bin_edge to initialize the value oh threshold
            th_in = bin_edge[0] + ((bin_edge[-1]-bin_edge[0])/2)

        #Fit
            popt, _ = curve_fit(gauss, bin_center, bin_height, p0 = [1, th_in, 3], bounds=(low, up))
            print(*popt)

        # #Save results in a txt file
            with open(f"th_fitresults_{the_vh}[{row_start},{row_stop} - {col_start},{col_stop}]({vclip}).txt", "w") as outf:
                print("#A#mean#sigma:", file=outf)
                print(*popt, file=outf)

        #Drawing the fit function with estimated parameters from the fit.
            xb = np.arange(bin_edge[0], bin_edge[-1], 0.005)
            plt.plot(xb, gauss(xb, *popt), "r-", label="fit")

            global mean
            mean = round(popt[1],3)
            sigma = round(popt[2],3)

        #TEXT
            plt.text(18, 120, f"mean = {mean}\nsigma = {sigma}", bbox = {"facecolor":"white"})#, ha="right", va="top") #CASCODE 170

        plt.title(subtitle)
        plt.suptitle("Threshold distribution")
        plt.xlabel("Threshold [DAC]")
        plt.ylabel("Pixels / bin")
        plt.grid(axis='y')
        pdf.savefig(); plt.clf()

        # THRESHOLD MAP
        # plt.pcolormesh(occupancy_edges[0], occupancy_edges[1], threshold_DAC.transpose(),
        #            rasterized=True)  # Necessary for quick save and view in PDF
        # plt.title(subtitle)
        # plt.suptitle("Threshold map")
        # plt.xlabel("Column")
        # plt.ylabel("Row")
        # cb = plt.colorbar()
        # cb.set_label("Threshold [DAC]")
        # pdf.savefig(); plt.clf()


        ##################################################

        histo2, charge_bins2, tot_bins2, _ = plt.hist2d(charge_dac, tot2, bins=[ch_lim, m],
            range=[[0, ch_lim], [0, m]], cmin=1, rasterized=True)
        np.nan_to_num(histo2, copy=False)

        plt.title(subtitle)
        plt.suptitle("ToT curve")
        plt.xlabel("Injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        cb = plt.colorbar()
        cb.set_label("Hits / bin")
        pdf.savefig(); plt.clf()

        charge_bins2 = (charge_bins2[:-1] + charge_bins2[1:])/2
        tot_bins2 = (tot_bins2[:-1] + tot_bins2[1:])/2
        tot_mean_temp2 = np.tile(tot_bins2, (len(charge_bins2),1))
        with np.errstate(invalid='ignore', divide='ignore'):
            tot_mean2 = np.sum(tot_mean_temp2*histo2,axis=1)/ np.sum(histo2, axis=1)

        #
        # # Removing point with no hits
        mask2 = np.isfinite(tot_mean2)
        tot_mean2 = tot_mean2[mask2]
        charge_bins2 = charge_bins2[mask2]

        low = [0 , -10, 0, 0]
        up = [0.2, 10,100, 80]
        param0 = [0.15, -2, 10, mean-10]

        def func(x,a,b,c,t):
            return np.where(x<mean, 0, np.maximum(0, a*x+b-(c/(x-t))))

        popt, pcov = curve_fit(func, charge_bins2, tot_mean2,p0 = param0, bounds=(low, up))#, maxfev=10000)

        perr = np.sqrt(np.diag(pcov))

        print(*popt)
        print(*perr)

        histo2, charge_bins2, tot_bins2, _ = plt.hist2d(charge_dac, tot2, bins=[ch_lim, m],
            range=[[0, ch_lim], [0, m]], cmin=1, rasterized=True)
        np.nan_to_num(histo2, copy=False)


        y = np.arange(mean-0.1, 500, 1)
        plt.plot(y, func(y,*popt), "r-", label="fit")

        plt.title(subtitle)
        plt.suptitle("ToT curve fit")
        plt.xlabel("True injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        cb = plt.colorbar()
        cb.set_label("Hits / bin")
        plt.text(10, 23, f"{vclip}\n\n"
            f"a = {ufloat(round(popt[0],3),round(perr[0], 3))}\n"
            f"b = {ufloat(round(popt[1],3),round(perr[1], 3))}\n"
            f"c = {ufloat(round(popt[2],3),round(perr[2], 3))}\n"
            f"t = {ufloat(round(popt[3],4),round(perr[3], 4))}",
            #f"t = {ufloat(np.format_float_scientific(popt[3]),np.format_float_scientific(perr[3]))}\n"
            #f"t = {np.format_float_scientific(popt[3], precision=3)}+/- {np.format_float_scientific(perr[3], precision=3)}",
            bbox = {"facecolor":"white"})
        pdf.savefig(); plt.clf()
        print


        #####################################################


        # TOT FIT
        hist = plt.hist2d(charge_dac, tot, bins=[ch_lim, m],
            range=[[0, ch_lim], [0, m]], cmin=1, rasterized=True)
        np.nan_to_num(histo, copy=False)

        # print(hist[0])
        # #print(hist[1])
        # #print(hist[2])
        # #print(hist[3])
        # tot = hist[0]
        # tot = tot[tot<18]
        # print(tot)
        #
        # hist = plt.hist2d(charge_dac, tot, bins=[ch_lim, m],
        #     range=[[0, ch_lim], [0, m]], cmin=1, rasterized=True)
        # np.nan_to_num(histo, copy=False)




        # low = [0 , -10, 0, 0]
        # up = [0.2, 10,100, 80]
        # param0 = [0.15, -0.5, 10, mean-10]

        low = [0 , -100, 0, 0]
        up = [0.2, 100,100, 80]
        param0 = [0.15, -0.5, 10, mean-10]

        def func(x,a,b,c,t):
            return np.where(x<mean, 0, np.maximum(0, a*x+b-(c/(x-t))))

        popt, pcov = curve_fit(func, charge_bins, tot_mean,p0 = param0, bounds=(low, up))#, maxfev=10000)

        # if frontend=="cascode":
        #     th_fit = mean       # Obtained by fitting THR distribution w/o saturation
        #     popt, pcov = curve_fit(func_casc, charge_bins, tot_mean,
        #         p0 = [0.15, -5, 10, 55],bounds=(low, up), maxfev=10000)
        # elif frontend=="normal":
        #     th_fit = mean
        #     popt, pcov = curve_fit(func_norm, charge_bins, tot_mean,
        #         p0 = [0.15, -5, 10, 10],bounds=(low, up), maxfev=10000)
        # else:
        #     logging.warning("Invalid argument for the front-end.")

        perr = np.sqrt(np.diag(pcov))

        print(*popt)
        print(*perr)

        y = np.arange(mean-0.1, 500, 1)
        plt.plot(y, func(y,*popt), "r-", label="fit")
        # if frontend=="cascode":
        #     plt.plot(y, func_casc(y, *popt), "r-", label="fit")
        # elif frontend=="normal":
        #     plt.plot(y, func_norm(y, *popt), "r-", label="fit")

        plt.title(subtitle)
        plt.suptitle("ToT curve fit")
        plt.xlabel("True injected charge [DAC]")
        plt.ylabel("ToT [25 ns]")
        cb = plt.colorbar()
        cb.set_label("Hits / bin")
        plt.text(10, 23, f"{vclip}\n\n"
            f"a ={ufloat(round(popt[0], 3), round(perr[0], 3))}\n"
            f"b = {ufloat(round(popt[1],3),round(perr[1],3))}\n"
            f"c = {ufloat(round(popt[2],3),round(perr[2], 3))}\n"
            #f"t = {ufloat(np.format_float_scientific(popt[3]),np.format_float_scientific(perr[3]))}\n"
            f"t = {np.format_float_scientific(popt[3], precision=3)}+/- {np.format_float_scientific(perr[3], precision=3)}",
            bbox = {"facecolor":"white"})
        pdf.savefig(); plt.clf()


        # TOT FIT ENLARGED
        # plt.hist2d(charge_dac, tot, bins=[250, m],
        #     range=[[-0.5, 249.5], [-0.5, m - 0.5]],
        #     cmin=1, rasterized=True)
        # np.nan_to_num(histo, copy=False)
        #
        # y = np.arange(th_fit-1, 500, 1)
        #
        # if frontend=="cascode":
        #     plt.plot(y, func_casc(y, *popt), "r-", label="fit")
        #     intercept = func_casc(250,popt[0], popt[1], popt[2], popt[3])
        # elif frontend=="normal":
        #     plt.plot(y, func_norm(y, *popt), "r-", label="fit")
        #     intercept = func_norm(250,popt[0], popt[1], popt[2], popt[3])
        #
        # perr = np.sqrt(np.diag(pcov))
        #
        # plt.xlim([0, 250])
        # plt.ylim([0,50])
        # plt.title(subtitle)
        # plt.suptitle("ToT curve fit")
        # plt.xlabel("True injected charge [DAC]")
        # plt.ylabel("ToT [25 ns]")
        # cb = plt.colorbar()
        # cb.set_label("Hits / bin")
        # plt.text(20, 40, f"a ={ufloat(round(popt[0], 3), round(perr[0], 3))}\n"
        #     f"b = {ufloat(round(popt[1],3),round(perr[1],3))}\n"
        #     f"c = {ufloat(round(popt[2],3),round(perr[2], 3))}\n"
        #     f"t = {ufloat(np.around(popt[3],3),round(popt[3], 3))}",
        #     bbox = {"facecolor":"white"})
        # pdf.savefig(); plt.clf()


def func_casc(x,a,b,c, t):
    v=62.87
    return np.where(x<v, 0, np.maximum(0, a*x+b-(c/(x-t))))

def func_norm(x,a,b,c, t):
    v= 52.35
    return np.where(x<v, 0, np.maximum(0, a*x+b-(c/(x-t))))

def gauss(x, A, mean, sigma):
    return  A * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


if __name__ == "__main__":

    #logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infile", nargs="*", help="The _threshold_scan_interpreted.h5 file(s)."
             " If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("-fe",type=str, required=True, help="Type of front end.")
    parser.add_argument("-f", "--overwrite", action="store_true", help="Overwrite plots when already present.")
    parser.add_argument("-s", "--shift", action="store_true", required=False, help="Shift value of charge if there's saturation of injection circuitry.")
    parser.add_argument("-c", "--cut", type=int, required=True, help="Cut on tot data to clean a little the plot.")
    args = parser.parse_args()
    logging.debug("Parser ok")

    files = []
    if args.infile:  # If anything was given on the command line
        logging.debug("We are in the first if")
        for pattern in args.infile:
            logging.debug("We are in the for")
            files.extend(glob.glob(pattern, recursive=True))
    else:
        files.extend(glob.glob("output_data/module_0/chip_0/*_threshold_scan_interpreted.h5"))

    for fp in tqdm(files):
        try:
            main(fp, args.overwrite, f"{args.fe}")
        except Exception:
            print(traceback.format_exc())
