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


def average(a, axis=None, weights=1, invalid=np.NaN):
    """Like np.average, but returns `invalid` instead of crashing if the sum of weights is zero."""
    return np.nan_to_num(np.sum(a * weights, axis=axis).astype(float) / np.sum(weights, axis=axis).astype(float), nan=invalid)


def main(infile, overwrite=False, frontend=None):
    outfile = os.path.splitext(infile)[0] + "_thfit.pdf"
    #if os.path.isfile(output_file) and not overwrite:
    #    return
    logging.info(f"Starting the analysis of {infile}")

    #Loading information on the chip configuration
    with tb.open_file(infile) as f, PdfPages(outfile) as pdf:
        cfg = get_config_dict(f)
        chip_serial_number = cfg["configuration_in.chip.settings.chip_sn"]
        plt.figure(figsize=(6.4,4.8))
        s=""
        param = ["IBIAS", "ITHR", "ICASN", "IDB", "ITUNE", "VRESET", "VCASC", "VCASP"]
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

        a = s.find("ICASN")
        icasn = s[a:a+10]
        c = s.find("ITHR")
        ithr = s[c:c+9]
        #Load hits
        hits = f.root.Dut[:]

############################## RETRIEVE TIMESTAMP ##########################

############################ pixel noisy ########################
        timestamp = hits["timestamp"]         #all timestamps

        noisy = hits[(hits["col"]==229)&(hits["row"]==484)]
        tot_noisy = noisy["te"]-noisy["le"]
        ts_noisy = noisy["timestamp"]
        noisy_null = hits[(hits["te"]-hits["le"])==0]
        tot_noisy_null = noisy_null["te"]-noisy_null["le"]
        ts_noisy_null = noisy_null["timestamp"] & 0x7f

        # plt.plot(ts_noisy_null, tot_noisy_null, "bo")
        # plt.xlabel("Timestamp")
        # plt.ylabel("ToT")
        # # plt.grid(axis='y')
        # pdf.savefig(); plt.clf()

        #noisy = hits[(hits["col"]==229)&(hits["row"]==484)]
        noisy = hits[(hits["col"]==227)&(hits["row"]==494)]
        tot_noisy = (noisy["te"]-noisy["le"]) & 0x7f
        ts_noisy = noisy["timestamp"]
        boh = (ts_noisy[1:]-ts_noisy[:-1])/640
        #print(boh[2500:2900])
        #print(tot_noisy[2500:2900])
        # noisy_null = hits[(hits["te"]-hits["le"])==0]
        # tot_noisy_null = noisy_null["te"]-noisy_null["le"]
        # ts_noisy_null = noisy_null["timestamp"] & 0x7f

        # plt.plot(ts_noisy_null, tot_noisy_null, "bo")
        # plt.xlabel("Timestamp")
        # plt.ylabel("ToT")
        # # plt.grid(axis='y')
        # pdf.savefig(); plt.clf()

        # tot_null = hits[(hits["te"]-hits["le"])==0]
        # ts_null = tot_null["timestamp"] & 0x7f
        # for i in range(0,len(ts_null),1):
        #     if tot_noisy[i]<0:
        #         tot_noisy[i]+=128
        # print(tot_noisy[2500:2900])
        # meh = np.where((tot_noisy[2500:2900]<13))
        # meh=meh[0]
        # meha = meh[1:]- meh[:-1]
        # print(meha)
        # plt.hist(boh)
        # plt.xlabel("Timestamp")
        # plt.ylabel("Counts")
        # plt.grid(axis='y')
        # pdf.savefig(); plt.clf()
        #
        # plt.hist(tot_noisy)
        # plt.xlabel("Timestamp")
        # plt.ylabel("Counts")
        # plt.grid(axis='y')
        # pdf.savefig(); plt.clf()

############################## TIMESTAMP #####################################
        # plt.hist2d(tot_noisy[1:],boh,bins=[128, 250], range=[[0, 80], [0, 18]], cmin=1)
        #
        # counto = np.count_nonzero(boh > 8.75)
        # plt.title(f"{ithr}, {icasn}. Pixel: 227, 494")
        # plt.suptitle("Timestamp vs ToT")
        # plt.xlabel("ToT[x 25 ns]")
        # plt.ylabel(r"$\Delta$Timestamp ["+r"$\mu$s]")
        # plt.grid(axis='y')
        # cb = plt.colorbar()
        # cb.set_label("Hits")
        # pdf.savefig(); plt.clf()

        #print(counto)
        #print(boh.max())


        # bohhh = bohh[bohh["te"]-bohh["le"]==0]
        # tot_noisy = (bohh["te"]-bohh["le"])  & 0x7f
        #
        # times_noisy_null = bohhh["timestamp"] & 0x7f
        # times = bohh["timestamp"] & 0x7f
        #
        #
        # plt.hist(times, bins=100)
        # plt.suptitle("Timestamp")
        # plt.xlabel("Timestamp")
        # plt.ylabel("Counts")
        # plt.grid(axis='y')
        # pdf.savefig(); plt.clf()

##################### pixel not noisy ##############################

        not_noisy = hits[(hits["col"]==227)&(hits["row"]==494)]
        # bohhh = bohh[bohh["te"]-bohh["le"]==0]
        # times_not = bohhh["timestamp"] & 0x7f
        #
        # plt.hist(times_not, bins=100)
        # plt.suptitle("Timestamp")
        # plt.xlabel("Timestamp")
        # plt.ylabel("Counts")
        # plt.grid(axis='y')
        # pdf.savefig(); plt.clf()

###################### timestamp vs tot noisy and not noisy ###################

        # plt.plot(times_noisy_null, bohh, "bo")
        # plt.suptitle("Timestamp vs tot")
        # plt.xlabel("Timestamp")
        # plt.ylabel("ToT")
        # plt.grid(axis='y')
        # pdf.savefig(); plt.clf()


        with np.errstate(all="ignore"):
            tot = (hits["te"] - hits["le"]) & 0x7f          #BITWISE AND

        logging.debug("Hits are saved and tot are calculated.")

        #Load information on injected charge and steps taken
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
        subtitle = f"VH = {the_vh}, VL = {start_vl}..{stop_vl} (step {step_vl})"
        logging.debug("Charge values are saved.")

        #BOH
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

        occu = np.amax(occupancy, axis=2)
        occu/=n_injections
        occupancy /= n_injections


################### PRINT PIXEL NOISY #######################################
        max_occu = np.max(occu)
        a = np.where(occu>=1.2)  #outndarray : An array with elements from x where condition is True, and elements from y elsewhere.
        noisy = len(a[0])

        logging.info(f"Pixels (col, row) with occupancy>1.1 are {noisy}:")
        text=[]
        for i in range(0, noisy , 1):
            #print(f"({a[0][i] + col_start}, {a[1][i] + row_start}), occupancy = {occu[a[0][i], a[1][i]]}")
            text.append(f"({a[0][i] + col_start}, {a[1][i] + row_start}), occupancy = {occu[a[0][i], a[1][i]]}\n")

        logging.info(f"Highest occupancy: {max_occu}")

################################### HITMAP ###################################

        plt.pcolormesh(occupancy_edges[0], occupancy_edges[1], occu.transpose(),
                    rasterized=True)  # Necessary for quick save and view in PDF
        plt.title(subtitle)
        plt.suptitle("Hits map")
        plt.xlabel("Column")
        plt.ylabel("Row")
        cb = plt.colorbar()
        cb.set_label("Occupancy")
        pdf.savefig(); plt.clf()

        if noisy!=0:
            rip = int(noisy/25)
            plt.title("Noisy pixel")
            for i in range(0, rip+1, 1):
                st = i*25
                sto = st+25
                text_def = "".join(text[st:sto])
                plt.annotate(split_long_text(f"{text_def}"), (0.01, 0.99), ha='left', va='top')
                plt.gca().set_axis_off()
                pdf.savefig(); plt.clf()


        occupancy_charges = occupancy_edges[2].astype(np.float32)
        occupancy_charges = (occupancy_charges[:-1] + occupancy_charges[1:]) / 2
        occupancy_charges = np.tile(occupancy_charges, (col_n, row_n, 1))

########################################## TOT ##################################

        #TOT Plot and fit
        m = 32 if tot.max() <= 32 else 128

        #Saving histo and data_bin in variables
        histo, charge_bins, tot_bins, _ = plt.hist2d(charge_dac, tot, bins=[250, m],
            range=[[-0.5, 249.5], [-0.5, m - 0.5]],
            cmin=1, rasterized=True)
        np.nan_to_num(histo, copy=False)

        #Taking the central value of each bin (chearge and tot)
        charge_bins = (charge_bins[:-1] + charge_bins[1:])/2
        tot_bins = (tot_bins[:-1] + tot_bins[1:])/2

        #Weighted mean of tot for eah charge_dac
        #Creating an array with the repetion of all tot_bins for each value of charge_dac
        #Then it does the average on the second axis, so y.
        tot_mean_a = np.tile(tot_bins, (len(charge_bins),1))
        with np.errstate(invalid='ignore', divide='ignore'):
            tot_mean = np.sum(tot_mean_a*histo,axis=1)/ np.sum(histo, axis=1)

        #Removing point with no hits
        mask = np.isfinite(tot_mean)
        tot_mean = tot_mean[mask]
        charge_bins = charge_bins[mask]

        #Fit
        # low = [-100 , -100, 1.2, 0]
        # up = [200, 200, 2, 200]
        # #popt, pcov = curve_fit(func, charge_bins, tot_mean, p0 = [0.18, -5, 1.3, 50], bounds=(low, up))
        #print(*popt)
        #plt.plot(charge_bins, func(charge_bins, *popt), "r-", label="fit")
        plt.title(subtitle)
        plt.suptitle("ToT curve fit")
        plt.xlabel("Injected charge [DAC]")
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
            with open(f"th_fitresults_{the_vh}[{row_start},{row_stop} - {col_start},{col_stop}]({ithr},{icasn}).txt", "w") as outf:
                print("#A#mean#sigma:", file=outf)
                print(*popt, file=outf)

        #Drawing the fit function with estimated parameters from the fit.
            xb = np.arange(bin_edge[0], bin_edge[-1], 0.005)
            plt.plot(xb, gauss(xb, *popt), "r-", label="fit")

            global mean
            mean = round(popt[1],3)
            sigma = round(popt[2],3)

        #TEXT
            plt.text(90, 201, f"mean = {mean}\nsigma = {sigma}", bbox = {"facecolor":"white"})#, ha="right", va="top") #CASCODE 170

        plt.title(subtitle)
        plt.suptitle("Threshold distribution")
        plt.xlabel("Threshold [DAC]")
        plt.ylabel("Pixels / bin")
        plt.grid(axis='y')
        pdf.savefig(); plt.clf()

        plt.pcolormesh(occupancy_edges[0], occupancy_edges[1], threshold_DAC.transpose(),
                   rasterized=True)  # Necessary for quick save and view in PDF
        plt.title(subtitle)
        plt.suptitle("Threshold map")
        plt.xlabel("Column")
        plt.ylabel("Row")
        cb = plt.colorbar()
        cb.set_label("Threshold [DAC]")
        pdf.savefig(); plt.clf()

#############################################################################
######################## NORMAL THRESHOLD ##################################

        if frontend=="all":
            occupancy_charges1 = occupancy_charges[0:224]
            threshold_DAC1 = average(occupancy_charges1, axis=2, weights=w[0:224])
            m11 = int(max(charge_dac_range[0], threshold_DAC1.min() - 2))
            m21 = int(min(charge_dac_range[1], threshold_DAC1.max() + 2))
            bin_height1, bin_edge1, _ = plt.hist(threshold_DAC1.reshape(-1), bins=30-m11, range=[m11, 30])
            bin_center1 = (bin_edge1[:-1]+bin_edge1[1:])/2# -0.5

        #######Fit
            #Range for fit parameters
            low = [0, 10, 0]
            up = [500, 200, 30]

            #Find the center value among bin_edge to initialize the value oh threshold
            th_in = bin_edge1[0] + ((bin_edge1[-1]-bin_edge1[0])/2)

            #Fit
            popt, _ = curve_fit(gauss, bin_center1, bin_height1, p0 = [1, th_in, 3], bounds=(low, up))
            print(*popt)

            #Save results in a txt file
            with open(f"th_fitresults_{the_vh}[{row_start},{row_stop} - 0,224]({ithr},{icasn}).txt", "w") as outf:
                print("#A#mean#sigma:", file=outf)
                print(*popt, file=outf)

            #Drawing the fit function with estimated parameters from the fit.
            xb = np.arange(bin_edge1[0], bin_edge1[-1], 0.005)
            plt.plot(xb, gauss(xb, *popt), "r-", label="fit")
            
            mean = round(popt[1],3)
            sigma = round(popt[2],3)


            plt.title(subtitle)
            plt.suptitle("Threshold distribution (Normal FE)")
            plt.xlabel("Threshold [DAC]")
            plt.ylabel("Pixels / bin")
            plt.text(bin_edge1[0]+5,55 , f"mean = {mean}\nsigma = {sigma}", bbox = {"facecolor":"white"})#, ha="right", va="top") #CASCODE 170
            plt.grid(axis='y')
            pdf.savefig(); plt.clf()

    ############################### CASCODE THRESHOLD ##############################
            occupancy_charges2 = occupancy_charges[224:448]
            threshold_DAC2 = average(occupancy_charges2, axis=2, weights=w[224:448])
            m12 = int(max(charge_dac_range[0], threshold_DAC2.min() - 2))
            m22 = int(min(charge_dac_range[1], threshold_DAC2.max() + 2))
            bin_height2, bin_edge2, _ = plt.hist(threshold_DAC2.reshape(-1), bins=30-m12, range=[m12, 30])
            bin_center2 = (bin_edge2[:-1]+bin_edge2[1:])/2# -0.5


            #######Fit
            #Range for fit parameters
            low = [0, 10, 0]
            up = [500, 200, 30]

            #Find the center value among bin_edge to initialize the value oh threshold
            th_in = bin_edge2[0] + ((bin_edge2[-1]-bin_edge2[0])/2)

            #Fit
            popt, _ = curve_fit(gauss, bin_center2, bin_height2, p0 = [1, th_in, 3], bounds=(low, up))
            print(*popt)

            #Save results in a txt file
            with open(f"th_fitresults_{the_vh}[{row_start},{row_stop} - 224,{col_stop}]({ithr},{icasn}).txt", "w") as outf:
                print("#A#mean#sigma:", file=outf)
                print(*popt, file=outf)

            #Drawing the fit function with estimated parameters from the fit.
            xb = np.arange(bin_edge2[0], bin_edge2[-1], 0.005)
            plt.plot(xb, gauss(xb, *popt), "r-", label="fit")
            mean = round(popt[1],2)
            sigma = round(popt[2],2)

            plt.title(subtitle)
            plt.suptitle("Threshold distribution (Cascode FE)")
            plt.xlabel("Threshold [DAC]")
            plt.ylabel("Pixels / bin")
            plt.text(bin_edge1[0]+5,55 , f"mean = {mean}\nsigma = {sigma}", bbox = {"facecolor":"white"})#, ha="right", va="top") #CASCODE 170
            plt.grid(axis='y')
            pdf.savefig(); plt.clf()


def func(x,a,b,c,t):
    return a*x + b - (c/(x-t))

def gauss(x, A, mean, sigma):
    return  A * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))



if __name__ == "__main__":

    #logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("infile", nargs="*", help="The _threshold_scan_interpreted.h5 file(s)."
             " If not given, looks in output_data/module_0/chip_0.")
    parser.add_argument("fe", type=str, help="Type of frontend : normal, cascode, all.")
    parser.add_argument("-f", "--overwrite", action="store_true", help="Overwrite plots when already present.")
    args = parser.parse_args()
    logging.debug("Parser ok")

    print(f"{args.fe}")

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
