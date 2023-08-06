#!/usr/bin/env python3
"""C_injection."""
import argparse
import glob
import os
import traceback
import itertools
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
from uncertainties import ufloat
import tables as tb
from tqdm import tqdm
from plot_utils_pisa import *
import sys
import math

FRONTENDS_TOT = [
    # a , b, c, t , name
    (0.12, 4, 200, 20, 'Normal'),
    (0.119, 1.4, 140, 40, 'Cascode'),
    (0.257, 3.2, 160, 17, 'HV Casc.'),
    (0.275, -5.7, 140, 42, 'HV')]

FRONTENDS_TOT2 = [
    # a , b, c, t , name
    (0.135, 0, 48, 'Normal'),
    (0.078, 10, 28, 'Cascode'),
    (0.257, 3.2, 17, 'HV Casc.'),
    (0.275, -5.7, 42, 'HV')]

def gauss(x, f0, mu0, sigma0):
    return (
    f0 * norm.pdf(x, mu0, sigma0)
    )

def main(th_file, peak_file, source, fe, overwrite=False):
    output_file =  f"C_inj_{source}.pdf"
    if os.path.isfile(output_file) and not overwrite:
        return

    with np.load(th_file) as th_data:
        with np.load(peak_file) as peak_data:
            thresholds = th_data['all_th']
            tot_peaks = peak_data['tot_peaks']

            # print(thresholds)
            # print(tot_peaks)
            # sys.exit()

            ################################################################
            #                    FOR EACH PIXELS                           #
            ################################################################
            # ranges = (512, 512, 32, 32)
            #

            # for (fc, lc, name), (a,b,c,t,n) in zip(FRONTENDS, FRONTENDS_TOT):
            #     if name==f"{fe}":
            #         print(name, fc, lc, a, b, c, t)
                    # B = b/(2*a)
                    # D = c/a
                # for col, row in tqdm(itertools.product(range(2), range(2), repeat=1), desc=f"Progress in {name}:"):
                #     if name=="Cascode":
                #         col+=224
                #     elif name=='HV Casc.':
                #         col+=448
                #     elif name=="HV":
                #         col+=480
                #     A = thresholds[col,row]/2
                #     C = tot_peaks[col,row]/(2*a)
                #
                #     tot_dac1 = (A-B+C) + math.sqrt((A+B-C)**2 + D)
                #     tot_dac2 = (A-B+C) - math.sqrt((A+B-C)**2 + D)
                #
                #
                #     print("\n")
                #     print(thresholds[col,row], A,B, C, D)
                #     print( tot_dac1, tot_dac2)

            #############################################################
            #             WITH MATRIX with t=threshold                  #
            #############################################################

            # B = b/(2*a)
            # D = c/a
            # a_mat = np.full((512,512), a)
            #
            # A_mat = thresholds/2
            # B_mat = np.full((512, 512), B)
            # D_mat = np.full((512, 512), D)
            # C_mat = tot_peaks/(2*a_mat)
            # alpha = np.full((512,512), 1616)
            # beta = np.full((512,512), 1781)
            #
            # tot_dac1 = (A_mat-B_mat+C_mat) + np.sqrt((A_mat+B_mat-C_mat)**2 + D_mat)
            # tot_dac2 = (A_mat-B_mat+C_mat) - np.sqrt((A_mat+B_mat-C_mat)**2 + D_mat)
            #
            # c_inj_alpha = alpha/tot_dac1
            # c_inj_beta = beta/tot_dac1
            #
            # np.savez_compressed(
            #     f"c_inj_{source}_{fe}.npz",
            #     c_inj_kalpha = c_inj_alpha,
            #     c_inj_kbeta = c_inj_beta)
            # print("\"*.npz\" file is created.")
            #
            #     # print(c_inj_alpha)
            #     # print(c_inj_beta)
            #
            # print(np.count_nonzero(~np.isnan(c_inj_alpha)))
            # min, max = np.nanmin(c_inj_alpha), np.nanmax(c_inj_alpha)
            # # print(np.nanmin(c_inj_alpha), np.nanmax(c_inj_alpha) )



            #############################################################
            #               WITH MATRIX with t from fit                 #
            #############################################################

            print(fe)

            for front in fe:
                for (fc, lc, name), (a,b,t,n) in zip(FRONTENDS, FRONTENDS_TOT2):
                    print(front, name)
                    if name==f"{front}":
                        print(name, fc, lc, a, b, t)
                        break

                print(fc,lc,name)

                if name=="Normal":
                    th_140 = 53.62
                elif name=="Cascode":
                    th_140 = 60.19
                elif name == "HV Casc.":
                    th_140 = 35.34
                elif name == "HV":
                    th_140 = 31.7

                print(f"Threshold:{th_140}")

                c = ((th_140)**2)*a-(a*t*th_140)+(b*th_140)-(t*b)
                print(f"c = {c}")

                A = t/2
                B = b/(2*a)
                D = c/a
                a_mat = np.full((512,512), a)

                #print(tot_peaks[448:450,0:12])


                #A_mat = thresholds/2
                A_mat = np.full((512,512), A)
                B_mat = np.full((512, 512), B)
                D_mat = np.full((512, 512), D)
                C_mat = tot_peaks/(2*a_mat)
                alpha = np.full((512,512), 1616)
                beta = np.full((512,512), 1781)




                #prendi solo quello maggiore della threshold
                tot_dac1 = (A_mat-B_mat+C_mat) + np.sqrt((A_mat+B_mat-C_mat)**2 + D_mat)
                tot_dac2 = (A_mat-B_mat+C_mat) - np.sqrt((A_mat+B_mat-C_mat)**2 + D_mat)



                c_inj_alpha = alpha/tot_dac1
                c_inj_beta = beta/tot_dac1

                np.savez_compressed(
                    f"c_inj_{source}_{front}_t.npz",
                    c_inj_kalpha = c_inj_alpha,
                    c_inj_kbeta = c_inj_beta)
                print("\"*.npz\" file is created.")

                # print(c_inj_alpha)
                # print(c_inj_beta)

                print(np.count_nonzero(~np.isnan(c_inj_alpha)))
                min, max = np.nanmin(c_inj_alpha), np.nanmax(c_inj_alpha)
                # print(np.nanmin(c_inj_alpha), np.nanmax(c_inj_alpha) )



                with PdfPages(f"c_inj_{source}_{front}.pdf") as pdf:
                    print(c_inj_alpha.shape)
                    temp = c_inj_alpha[fc:lc+1]
                    dat =temp[~np.isnan(temp)]
                    print(dat.shape)



                    #print(c_inj_beta.shape)
                    temp2 = c_inj_beta[fc:lc+1]
                    dat2 =temp2[~np.isnan(temp2)]
                    #print(dat2.shape)

                    # dat2 = c_inj_beta[~np.isnan(c_inj_beta)]
                    # print(fc, lc)
                    # print(len(dat[fc:lc+1]))
                    ##################################################################
                    #               ALPHA

                    #plt.hist(c_inj_alpha[fc:lc+1].reshape(-1),bins=20, range=[int(min-.5),int(max+1.5)], label=fe)
                    if name== "HV Casc.":
                        bin_height, bin_edge,_ = plt.hist(dat.reshape(-1), range=[14, 24], label=front)
                    elif name=="HV":
                        bin_height, bin_edge,_ = plt.hist(dat.reshape(-1),range = [12,18], label=front)


                    print(bin_height)

                    plt.xlabel("C_inj_alpha [$e^{-}$/DAC]")
                    plt.ylabel("Counts")
                    plt.title(f"Injection capacitance of {front} flavor.")
                    plt.legend()
                    plt.grid()
                    pdf.savefig(); plt.clf()

                    #FIT
                    bin_center = (bin_edge[:-1]+bin_edge[1:])/2

                    entries = np.sum(bin_height)
                    popt, pcov = curve_fit(gauss, bin_center, bin_height, p0=[0.2*entries, 13, 2])
                    perr = np.sqrt(np.diag(pcov))
                    x = np.arange(bin_edge[0], bin_edge[-1], 0.005)

                    if name=="HV Casc.":
                        bin_height, bin_edge,_ = plt.hist(dat.reshape(-1),range=[14, 24], label=front)
                    elif name=="HV":
                        bin_height, bin_edge,_ = plt.hist(dat.reshape(-1),range = [12,18], label=front)

                    plt.plot(x, gauss(x, *popt), "r-", label=f"fit {name}:\nmean={ufloat(round(popt[1], 3), round(perr[1],3))}\nsigma={ufloat(round(popt[2], 3), round(perr[2],3))}")
                    plt.xlabel("C_inj_alpha [$e^{-}$/DAC]")
                    plt.ylabel("Counts")
                    plt.title(f"Injection capacitance of {front} flavor.")
                    plt.legend()
                    plt.grid()
                    pdf.savefig();
                    plt.savefig(f"c_inj_alpha_fe_{name}.png"); plt.clf()

                    for m, s, n in zip(popt, np.sqrt(pcov.diagonal()), ["f0", "mu0", "sigma0"]):
                        print(f"{n:>10s} = {ufloat(m,s,t)}")




        #######################################################################
        #                       BETA

        #plt.hist(c_inj_beta[fc:lc+1].reshape(-1),bins=20, range=[int(min-.5),int(max+1.5)], label=fe)
        # if name=="Cascode":
        #     bin_height, bin_edge,_ = plt.hist(dat2.reshape(-1),range=[7, 11],label=fe)
        # elif name=="Normal":
        #     bin_height, bin_edge,_ = plt.hist(dat2.reshape(-1),range = [8,12],label=fe)
        #
        # plt.xlabel("C_inj_beta [$e^{-}$/DAC]")
        # plt.ylabel("Counts")
        # plt.title(f"Injection capacitance of {fe} flavor.")
        # plt.legend()
        # plt.grid()
        # pdf.savefig(); plt.clf()
        #
        #
        # #FIT
        # bin_center = (bin_edge[:-1]+bin_edge[1:])/2
        # entries = np.sum(bin_height)
        # popt, pcov = curve_fit(gauss, bin_center, bin_height, p0=[0.2*entries, 13, 1])
        # perr = np.sqrt(np.diag(pcov))
        # x = np.arange(bin_edge[0], bin_edge[-1], 0.005)
        #
        # if name=="Cascode":
        #     bin_height, bin_edge,_ = plt.hist(dat2.reshape(-1),range=[7, 11],label=fe)
        # elif name=="Normal":
        #     bin_height, bin_edge,_ = plt.hist(dat2.reshape(-1),range = [8,12],label=fe)
        #
        # plt.plot(x, gauss(x, *popt), "r-", label=f"fit {name}:\nmean={ufloat(round(popt[1], 3), round(perr[1],3))}\nsigma={ufloat(round(popt[2], 3), round(perr[2],3))}")
        # plt.xlabel("C_inj_beta [$e^{-}$/DAC]")
        # plt.ylabel("Counts")
        # plt.title(f"Injection capacitance of {fe} flavor.")
        # plt.legend()
        # plt.grid()
        # pdf.savefig();
        # plt.savefig(f"c_inj_alpha_fe_{name}.png"); plt.clf()
        #
        # for m, s, n in zip(popt, np.sqrt(pcov.diagonal()), ["f0", "mu0", "sigma0"]):
        #     print(f"{n:>10s} = {ufloat(m,s,t)}")

        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument( "-th", "--thresholds",
        help="The npz file with the values of the thresholds for each pixel.")
    parser.add_argument("-p", "--peaks",
        help="The npz file with the values of the tot_peak for each pixel.")
    parser.add_argument("-s", "--source",
        help="Source used for the data acquisition.")
    parser.add_argument("-fe", "--frontend", nargs="+",
        help="Frontend.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    args = parser.parse_args()

    print(args.frontend)

    try:
        main(args.thresholds,args.peaks, args.source,args.frontend, args.overwrite)
    except Exception:
        print(traceback.format_exc())
