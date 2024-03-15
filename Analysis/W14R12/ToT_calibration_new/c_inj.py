#!/usr/bin/env python3
"""C_injection."""

# python3 c_inj.py -th "all_thresholds_norm.npz" -p "tot_fe_peaks.npz" -s Fe -fe Normal -f


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


# FRONTENDS_TOT2 = [
#     # a , b, c, t , name
#     (0.135, 0, 48, 'Normal'),
#     (0.078, 10, 28, 'Cascode'),
#     (0.257, 3.2, 17, 'HV Casc.'),
#     (0.275, -5.7, 42, 'HV')]

FRONTENDS_TOT0 = [
    # a , b, c, t , name
    (0.12, 4, 200, 20, 'Normal'),
    (0.119, 1.4, 140, 40, 'Cascode'),
    (0.257, 3.2, 160, 17, 'HV Casc.'),
    (0.275, -5.7, 140, 42, 'HV')]


FRONTENDS_TOT1 = [
    # a , b, c, t , name
    (0.116, 4, 250, 22, 'Normal'),
    (0.119, 1.5, 140, 40, 'Cascode'),
    (0.247, 5.7, 330, 3, 'HV Casc.'),
    (0.269, 4.2, 260, 1, 'HV')]


FRONTENDS_TOT2 = [
    # a , b, c, t , name
    (0.138, -0.8, 60, 42.3, 'Normal'),
    (0.126, -0.2, 87, 47, 'Cascode'),
    (0.2657, 1.09, 67, 28.3, 'HV Casc.'),
    (0.2839, 0.3, 45, 27, 'HV')]


FRONTENDS_TOT3 = [
    # a , b, c, t , name
    (0.146, -2.2, 25, 48.9, 'Normal'),
    (0.138, -2.5, 28, 56, 'Cascode'),
    (0.2685, 0.6, 55, 29.7, 'HV Casc.'),
    (0.2839, 0.3, 58, 25.5, 'HV')]


FRONTENDS_TOT4 = [
    # a , b, c, t , name
    (0.1470, -2.72, 16, 50.2, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

FRONTENDS_TOT_LAST = [
    # a , b, c, t , name
    (0.145, -2.2, 24, 49.1, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

##############################################################################

FRONTENDS_TOT_FIRST = [
    # a , b, c, t , name
    (0.1228, 2.17, 132, 34.8, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

FRONTENDS_TOT_SECOND = [
    # a , b, c, t , name
    (0.131, 0.2, 84, 39.5, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

FRONTENDS_TOT_THIRD_C = [
    # a , b, c, t , name
    (0.142, -1.8, 29, 48.4, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

FRONTENDS_TOT_THIRD_T = [
    # a , b, c, t , name
    (0.1271, 1.11, 81, 43.17, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

########################################

FRONTENDS_TOT_FIRST2 = [
    # a , b, c, t , name
    (0.1233, 2.01, 123, 36.3, 'Normal'),
    (0.1298, -1.02, 33.6, 59., 'Cascode'),
    (0.2597, 2.63, 126, 21.1, 'HV Casc.'),
    (0.2799, 1.49, 89, 20.8, 'HV')]

FRONTENDS_TOT_THIRD_T2 = [
    # a , b, c, t , name
    (0.1289, 0.77, 70, 44.17, 'Normal'),
    (0.1257, -0.14, 62, 52.52, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

##############################################
FRONTENDS_TOT_FOURTH_T = [
    # a , b, c, t , name
    (0.1395, -1.5, 20, 49.94, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]

FRONTENDS_TOT_FOURTH_C = [
    # a , b, c, t , name
    (0.1435, -2.32, 19, 49.9, 'Normal'),
    (0.1402, -3.21, 17, 57.6, 'Cascode'),
    (0.2765, -1.04, 28, 31.9, 'HV Casc.'),
    (0.291, -1.12, 28, 28.01, 'HV')]



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

            print("Check:\n")

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

            for (fc, lc, name), (a,b,c,t,n) in zip(FRONTENDS, FRONTENDS_TOT_FIRST2):
                if name==f"{fe}":
                    print(name, fc, lc, a, b,c, t)
                    break

            print(f"Flavour: {fc, lc, name, a, b, c, t}")

            if name=="Normal":
                #th_140 = 53.62
                th_140 = 53.37
            elif name=="Cascode":
                #th_140 = 60.19
                th_140 = 60.8

            print(f"Threshold:{th_140}")

            #c1 = ((th_140)**2)*a-(a*t*th_140)+(b*th_140)-(t*b)
            c1 = (th_140-t)*(a*th_140+b)
            print(f"Value of C: {c1, c}")


            A = t/2
            B = b/(2*a)
            D = c/a
            a_mat = np.full((512,512), a)

            #A_mat = thresholds/2
            A_mat = np.full((512,512), A)
            B_mat = np.full((512, 512), B)
            D_mat = np.full((512, 512), D)
            C_mat = tot_peaks/(2*a_mat)             # [fc:lc+1, :]
            alpha = np.full((512,512), 1616)
            #beta = np.full((512,512), 1781)


            #prendi solo quello maggiore della threshold
            tot_dac1 = (A_mat-B_mat+C_mat) + np.sqrt((A_mat+B_mat-C_mat)**2 + D_mat)
            #tot_dac2 = (A_mat-B_mat+C_mat) - np.sqrt((A_mat+B_mat-C_mat)**2 + D_mat)

            c_inj_alpha = alpha/tot_dac1
            #c_inj_beta = beta/tot_dac1

            np.savez_compressed(
                f"c_inj_{source}_{fe}_t.npz",
                c_inj_kalpha = c_inj_alpha)#,
                #c_inj_kbeta = c_inj_beta)
            print("\"*.npz\" file is created.")

            print(f"K factors:{c_inj_alpha}")
            #print(c_inj_beta)

            print(np.count_nonzero(~np.isnan(c_inj_alpha)))
            min, max = np.nanmin(c_inj_alpha), np.nanmax(c_inj_alpha)
            # print(np.nanmin(c_inj_alpha), np.nanmax(c_inj_alpha) )


            ###################################################################
            # NORMAL
            for (fc, lc, name), (a,b,c,t,n) in zip(FRONTENDS, FRONTENDS_TOT_FIRST2):
                if name==f"{fe}":
                    print(name, fc, lc, a, b,c, t)
                    break

            print(f"Flavour: {fc, lc, name, a, b, c, t}")

            if name=="Normal":
                #th_140 = 53.62
                th_140 = 53.37
            elif name=="Cascode":
                #th_140 = 60.19
                th_140 = 60.8

            print(f"Threshold:{th_140}")
            c1 = (th_140-t)*(a*th_140+b)        #c1 = ((th_140)**2)*a-(a*t*th_140)+(b*th_140)-(t*b)
            print(f"Value of C: {c1, c}")

            if name=="Normal":
                print("Siamo in Normal")
                A1 = t/2
                B1 = b/(2*a)
                D1 = c/a
                a_mat1 = np.full((224,512), a)

                #A_mat = thresholds/2
                A_mat1 = np.full((224,512), A1)
                B_mat1 = np.full((224, 512), B1)
                D_mat1 = np.full((224, 512), D1)
                C_mat1 = tot_peaks[0:224,:]/(2*a_mat1)
                alpha = np.full((224,512), 1616)
                #beta = np.full((512,512), 1781)


                #prendi solo quello maggiore della threshold
                tot_dac1_2 = (A_mat1-B_mat1+C_mat1) + np.sqrt((A_mat1+B_mat1-C_mat1)**2 + D_mat1)
                #tot_dac2 = (A_mat-B_mat+C_mat) - np.sqrt((A_mat+B_mat-C_mat)**2 + D_mat)

                c_inj_alpha1 = alpha/tot_dac1_2
                #c_inj_beta = beta/tot_dac1

                # np.savez_compressed(
                #     f"c_inj_{source}_{fe}_t.npz",
                #     c_inj_kalpha = c_inj_alpha)#,
                #     #c_inj_kbeta = c_inj_beta)
                # print("\"*.npz\" file is created.")

                print(f"K factors:{c_inj_alpha1}")
                #print(c_inj_beta)

                print(np.array_equal(c_inj_alpha[0:224, 0:512], c_inj_alpha1[0:224, 0:512], equal_nan=True))
                # comparison = c_inj_alpha[0:224,0:512] == c_inj_alpha1[0:224, 0:512]
                # print(comparison)
                # x=np.where(comparison==False)
                # print(x)


                print(np.count_nonzero(~np.isnan(c_inj_alpha)))
                min, max = np.nanmin(c_inj_alpha), np.nanmax(c_inj_alpha)
                # print(np.nanmin(c_inj_alpha), np.nanmax(c_inj_alpha) )

            ##################################################################
            # Cascode
            for (fc, lc, name), (a,b,c,t,n) in zip(FRONTENDS, FRONTENDS_TOT_FIRST2):
                if name==f"{fe}":
                    print(name, fc, lc, a, b,c, t)
                    break

            print(f"Flavour: {fc, lc, name, a, b, c, t}")

            if name=="Normal":
                #th_140 = 53.62
                th_140 = 53.37
            elif name=="Cascode":
                #th_140 = 60.19
                th_140 = 60.8

            print(f"Threshold:{th_140}")
            c1 = (th_140-t)*(a*th_140+b)        #c1 = ((th_140)**2)*a-(a*t*th_140)+(b*th_140)-(t*b)
            print(f"Value of C: {c1, c}")


            if name=="Cascode":
                print("Siamo in Cascode")
                A1 = t/2
                B1 = b/(2*a)
                D1 = c/a
                a_mat1 = np.full((224,512), a)

                #A_mat = thresholds/2
                A_mat1 = np.full((224,512), A1)
                B_mat1 = np.full((224, 512), B1)
                D_mat1 = np.full((224, 512), D1)
                C_mat1 = tot_peaks[224:448,:]/(2*a_mat1)
                alpha = np.full((224,512), 1616)


                #prendi solo quello maggiore della threshold
                tot_dac1_2 = (A_mat1-B_mat1+C_mat1) + np.sqrt((A_mat1+B_mat1-C_mat1)**2 + D_mat1)

                c_inj_alpha1 = alpha/tot_dac1_2
                print(f"K factors:{c_inj_alpha1}")

                print(np.array_equal(c_inj_alpha[224:448, 0:512], c_inj_alpha1[0:224, 0:512], equal_nan=True))
                # comparison = c_inj_alpha[224:448,0:512] == c_inj_alpha1[0:224, 0:512]
                # print(comparison)
                # x=np.where(comparison==False)
                # print(x)


                print(np.count_nonzero(~np.isnan(c_inj_alpha)))
                min, max = np.nanmin(c_inj_alpha), np.nanmax(c_inj_alpha)
                # print(np.nanmin(c_inj_alpha), np.nanmax(c_inj_alpha) )






    with PdfPages(f"c_inj_{source}_{fe}.pdf") as pdf:
        print(c_inj_alpha.shape)
        temp = c_inj_alpha[fc:lc+1]
        dat =temp[~np.isnan(temp)]
        print(dat.shape)

        # print(c_inj_beta.shape)
        # temp2 = c_inj_beta[fc:lc+1]
        # dat2 =temp2[~np.isnan(temp2)]
        # print(dat2.shape)

        # dat2 = c_inj_beta[~np.isnan(c_inj_beta)]
        # print(fc, lc)
        # print(len(dat[fc:lc+1]))
        ##################################################################
        #               ALPHA

        #plt.hist(c_inj_alpha[fc:lc+1].reshape(-1),bins=20, range=[int(min-.5),int(max+1.5)], label=fe)
        if name== "Cascode":
            bin_height, bin_edge,_ = plt.hist(dat.reshape(-1), range=[7.7, 10.3], label=fe)
        elif name=="Normal":
            bin_height, bin_edge,_ = plt.hist(dat.reshape(-1),range = [8,10], label=fe)

        # print(bin_height)

        plt.xlabel("K [$e^{-}$/DAC]")
        plt.ylabel("Counts")
        plt.title(f"Conversion factors K of {fe} flavor.")
        plt.legend()
        plt.grid()
        pdf.savefig(); plt.clf()

        #FIT
        bin_center = (bin_edge[:-1]+bin_edge[1:])/2

        entries = np.sum(bin_height)
        popt, pcov = curve_fit(gauss, bin_center, bin_height, p0=[0.2*entries, 13, 2])
        perr = np.sqrt(np.diag(pcov))
        x = np.arange(bin_edge[0], bin_edge[-1], 0.005)

        if name=="Cascode":
            bin_height, bin_edge,_ = plt.hist(dat.reshape(-1),range=[7.7, 10.3], label=fe)
        elif name=="Normal":
            bin_height, bin_edge,_ = plt.hist(dat.reshape(-1),range = [8,10], label=fe)

        plt.plot(x, gauss(x, *popt), "r-", label=f"fit {name}:\nmean={ufloat(round(popt[1], 3), round(perr[1],3))}\nsigma={ufloat(round(popt[2], 3), round(perr[2],3))}")
        plt.xlabel("K [$e^{-}$/DAC]")
        plt.ylabel("Counts")
        plt.title(f"Conversion factors of {fe} flavor.")
        plt.legend()
        plt.grid()
        pdf.savefig();
        plt.savefig(f"k_fe_{name}.png");plt.clf()

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
        # pdf.savefig(); plt.clf()
        #
        # for m, s, n in zip(popt, np.sqrt(pcov.diagonal()), ["f0", "mu0", "sigma0"]):
        #     print(f"{n:>10s} = {ufloat(m,s,t)}")
        #
        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument( "-th", "--thresholds",
        help="The npz file with the values of the thresholds for each pixel.")
    parser.add_argument("-p", "--peaks",
        help="The npz file with the values of the tot_peak for each pixel.")
    parser.add_argument("-s", "--source",
        help="Source used for the data acquisition.")
    parser.add_argument("-fe", "--frontend",
        help="Frontend.")
    parser.add_argument("-f", "--overwrite", action="store_true",
                        help="Overwrite plots when already present.")
    args = parser.parse_args()


    try:
        main(args.thresholds,args.peaks, args.source,args.frontend, args.overwrite)
    except Exception:
        print(traceback.format_exc())
