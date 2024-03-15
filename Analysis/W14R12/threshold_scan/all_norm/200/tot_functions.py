"""Functions."""

import argparse
import glob
from itertools import chain
import os
import sys
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
import copy
import sympy
import math



# C ERROR FOR THIRD AND FOURTH FIT

def c_err_third(pcov1,thr,dthr,a1,b1,t1):

    thr=float(thr)

    th, a, b, t = sympy.symbols('th,a,b,t')
    c = (th-t)*(a*th+b)
    c_ev = c.subs({a:a1, b:b1, t:t1, th:thr})
    dca = sympy.diff(c, a)
    dcb = sympy.diff(c, b)
    dct = sympy.diff(c, t)
    dcth = sympy.diff(c, th)
    dcth_ev = dcth.subs({a:a1, b:b1, t:t1, th:thr})

    ############################

    # ARRAY
    M_row = sympy.Matrix([[dca, dcb, dct]])                 #ROW
    M_col = sympy.Matrix([dca, dcb, dct])                   #COLUMN
    M1 = M_row.subs({a:a1, b:b1, t:t1, th:thr})
    M2 = M_col.subs({a:a1, b:b1, t:t1, th:thr})

    # TO NUMPY
    M3 = sympy.matrix2numpy(M1) #riga
    M4 = sympy.matrix2numpy(M2) #colonna

    # ROW-COLUMN PRODUCT
    M5 = np.dot(M3,pcov1)
    M6 = np.dot(M5,M4)
    dz1 = math.sqrt(M6)
    print(f"dz1 = {dz1}")

    M7 = M6 + ((dcth_ev*dthr)**2)
    dz2 = math.sqrt(M7)
    print(f"dz2 = {dz2}")

    ######################################
    # da = pcov1[0,0]*(dca)**2
    # db = pcov1[1,1]*(dcb)**2
    # dt = pcov1[2,2]*(dct)**2
    # dth = (dcth**2)*(dthr**2)
    #
    # dab = 2*dca*dcb*pcov1[0,1]
    # dat = 2*dca*dct*pcov1[0,2]
    # dbt = 2*dcb*dct*pcov1[1,2]
    #
    # d_2c = da + db + dt + dth + dab + dat + dbt
    # d_c = sympy.sqrt(d_2c)
    # dc_res = d_c.subs({a:a1, b:b1, t:t1, th:thr})
    #
    # return dc_res
    #####################################
    return dz2, c_ev



def c_err_fourth(pcov1,thr,dthr,a1,b1,t1, errt):

    thr=float(thr)

    th, a, b, t = sympy.symbols('th,a,b,t')
    c = (th-t)*(a*th+b)
    c_ev = c.subs({a:a1, b:b1, t:t1, th:thr})
    dca = sympy.diff(c, a)
    dcb = sympy.diff(c, b)
    dct = sympy.diff(c, t)
    dcth = sympy.diff(c, th)
    dcth_ev = dcth.subs({a:a1, b:b1, t:t1, th:thr})
    dct_ev = dct.subs({a:a1, b:b1, t:t1, th:thr})

    ############################

    # ARRAY
    M_row = sympy.Matrix([[dca, dcb]])                 #ROW
    M_col = sympy.Matrix([dca, dcb])                   #COLUMN
    M1 = M_row.subs({a:a1, b:b1, t:t1, th:thr})
    M2 = M_col.subs({a:a1, b:b1, t:t1, th:thr})

    # TO NUMPY
    M3 = sympy.matrix2numpy(M1) #riga
    M4 = sympy.matrix2numpy(M2) #colonna

    # ROW-COLUMN PRODUCT
    M5 = np.dot(M3,pcov1)
    M6 = np.dot(M5,M4)
    dz1 = math.sqrt(M6)
    print(f"dz1 = {dz1}")

    M7 = M6 + ((dcth_ev*dthr)**2) + ((dct_ev*errt)**2)
    dz2 = math.sqrt(M7)
    print(f"dz2 = {dz2}")

    #########################################
    # da = pcov1[0,0]*(dca)**2
    # db = pcov1[1,1]*(dcb)**2
    # dt = (errt**2)*(dct)**2
    # dth = (dcth**2)*(dthr**2)
    #
    # dab = 2*dca*dcb*pcov1[0,1]
    #
    # d_2c = da + db + dt + dth + dab
    # d_c = sympy.sqrt(d_2c)
    # dc_res = d_c.subs({a:a1, b:b1, t:t1, th:thr})

    #return dc_res
    #########################################
    return dz2, c_ev






# THRESHOLD DISPERSION FROM TOT CALIBRATION CURVE

def th_disp(a1, b1, c1, t1, pcov1):
    a, b, c, t = sympy.symbols('a, b, c, t')

    x_thr =  (t/2) - (b/(2*a)) + sympy.sqrt((t/2 + (b/(2*a)))**2 + c/a)
    x_thr2 =  (t/2) - (b/(2*a)) - sympy.sqrt((t/2 + (b/(2*a)))**2 + c/a)
    x_thr_ev = x_thr.subs({a:a1, b:b1, t:t1, c:c1})
    x_thr2_ev = x_thr2.subs({a:a1, b:b1, t:t1, c:c1})

    # PARTIAL DERIVATIVES

    dxa = sympy.diff(x_thr, a)
    dxb = sympy.diff(x_thr, b)
    dxc = sympy.diff(x_thr, c)
    dxt = sympy.diff(x_thr, t)


    ##############################

    dxc_ev = dxc.subs({a:a1, b:b1, t:t1, c:c1})

    # ARRAY
    M_row = sympy.Matrix([[dxa, dxb,dxc, dxt]])   #ROW
    M_col = sympy.Matrix([dxa,dxb,dxc,dxt])        #COLUMN
    M1 = M_row.subs({a:a1, b:b1, t:t1, c:c1})
    M2 = M_col.subs({a:a1, b:b1, t:t1, c:c1})

    # TO NUMPY
    M3 = sympy.matrix2numpy(M1) #riga
    M4 = sympy.matrix2numpy(M2) #colonna

    # ROW-COLUMN PRODUCT
    M5 = np.dot(M3,pcov1)
    M6 = np.dot(M5,M4)
    dz1 = math.sqrt(M6)
    #print(f"dz1 = {dz1}")

    return dz1, x_thr_ev, x_thr2_ev

def th_disp_third(a1, b1, c1, t1, pcov1, err_c21):
    a, b, c, t = sympy.symbols('a, b, c, t')

    x_thr =  (t/2) - (b/(2*a)) + sympy.sqrt((t/2 + (b/(2*a)))**2 + c/a)
    x_thr2 =  (t/2) - (b/(2*a)) - sympy.sqrt((t/2 + (b/(2*a)))**2 + c/a)
    x_thr_ev = x_thr.subs({a:a1, b:b1, t:t1, c:c1})
    x_thr2_ev = x_thr2.subs({a:a1, b:b1, t:t1, c:c1})

    c1 = float(c1)

    # PARTIAL DERIVATIVES

    dxa = sympy.diff(x_thr, a)
    dxb = sympy.diff(x_thr, b)
    dxc = sympy.diff(x_thr, c)
    dxt = sympy.diff(x_thr, t)


    ##############################

    dxc_ev = dxc.subs({a:a1, b:b1, t:t1, c:c1})

    # ARRAY
    M_row = sympy.Matrix([[dxa, dxb, dxt]])   #ROW
    M_col = sympy.Matrix([dxa,dxb,dxt])        #COLUMN
    M1 = M_row.subs({a:a1, b:b1, t:t1, c:c1})
    M2 = M_col.subs({a:a1, b:b1, t:t1, c:c1})

    # TO NUMPY
    M3 = sympy.matrix2numpy(M1) #riga
    M4 = sympy.matrix2numpy(M2) #colonna

    # ROW-COLUMN PRODUCT
    M5 = np.dot(M3,pcov1)
    M6 = np.dot(M5,M4)
    dz1 = math.sqrt(M6)
    print(f"dz1 = {dz1}")

    M7 = M6 + ((dxc_ev*err_c21)**2)
    dz2 = math.sqrt(M7)
    print(f"dz2 = {dz2}")

    ##################################

    # print(dxa)
    # print(dxb)
    # print(dxc)
    # print(dxt)

    # dab = 2*dxa*dxb*pcov_lin1[0,1]
    #
    # da = (dxa**2)*pcov1[0,0]
    # db = (dxb**2)*pcov1[1,1]
    # dc = (dxc**2)*(err_c21**2)
    # dt = (dxt**2)*(pcov1[2,2])
    #
    # d_2c = da + db + dt + dc + dab
    # d_c = sympy.sqrt(d_2c)
    # dc_res = d_c.subs({a:a1, b:b1, t:t1, c:c1})

    #z = x_thr.subs({ a:a1, b:b1, t:t1, c:c1})
    # dz = dy.subs({x:160, a:a1, b:b1, t:t1, c:c1})


    return dz2, x_thr_ev, x_thr2_ev


def th_disp_third_woc(a1, b1, t1, pcov1, th1, dth):
    a, b, t, th = sympy.symbols('a, b, t, th')

    th1=float(th1)

    x_thr =  (t/2) - (b/(2*a)) + sympy.sqrt((t/2 + (b/(2*a)))**2 + ((th-t)*(a*th+b))/a)
    x_thr2 =  (t/2) - (b/(2*a)) - sympy.sqrt((t/2 + (b/(2*a)))**2 + ((th-t)*(a*th+b))/a)
    x_thr_ev = x_thr.subs({a:a1, b:b1, t:t1, th:th1})
    x_thr2_ev = x_thr2.subs({a:a1, b:b1, t:t1, th:th1})

    # PARTIAL DERIVATIVES

    dxa = sympy.diff(x_thr, a)
    dxb = sympy.diff(x_thr, b)
    dxt = sympy.diff(x_thr, t)
    dxth = sympy.diff(x_thr, th)
    dxth_ev = dxth.subs({a:a1, b:b1, t:t1, th:th1})


    ##############################

    # ARRAY
    M_row = sympy.Matrix([[dxa,dxb,dxt]])   #ROW
    M_col = sympy.Matrix([dxa,dxb,dxt])        #COLUMN
    M1 = M_row.subs({a:a1, b:b1, t:t1, th:th1})
    M2 = M_col.subs({a:a1, b:b1, t:t1, th:th1})

    # TO NUMPY
    M3 = sympy.matrix2numpy(M1) #riga
    M4 = sympy.matrix2numpy(M2) #colonna

    # ROW-COLUMN PRODUCT
    M5 = np.dot(M3,pcov1)
    M6 = np.dot(M5,M4)
    dz1 = math.sqrt(M6)
    print(f"dz1 = {dz1}")

    M7 = M6 + ((dxth_ev*dth)**2)
    dz2 = math.sqrt(M7)
    print(f"dz2 = {dz2}")

    return dz2, x_thr_ev, x_thr2_ev





def th_disp_fourth(a1, b1, c1, t1, pcov_lin1, pcov1, err_c21):

    a, b, c, t = sympy.symbols('a, b, c, t')
    x_thr =  (t/2) - (b/(2*a)) + sympy.sqrt((t/2 + (b/(2*a)))**2 + c/a)
    x_thr2 =  (t/2) - (b/(2*a)) - sympy.sqrt((t/2 + (b/(2*a)))**2 + c/a)
    x_thr_ev = x_thr.subs({a:a1, b:b1, t:t1, c:c1})
    x_thr2_ev = x_thr2.subs({a:a1, b:b1, t:t1, c:c1})

    c1 = float(c1)

    dxa = sympy.diff(x_thr, a)
    dxb = sympy.diff(x_thr, b)
    dxc = sympy.diff(x_thr, c)
    dxt = sympy.diff(x_thr, t)

    #########################################
    # dab = 2*dxa*dxb*pcov_lin1[0,1]
    #
    # da = (dxa**2)*pcov_lin1[0,0]
    # db = (dxb**2)*pcov_lin1[1,1]
    # dc = (dxc**2)*(err_c21**2)
    # dt = (dxt**2)*(pcov1[0,0])
    #
    # d_2c = da + db + dt + dc + dab
    # d_c = sympy.sqrt(d_2c)
    # dc_res = d_c.subs({a:a1, b:b1, t:t1, c:c1})
    ##################################

    dxc_ev = dxc.subs({a:a1, b:b1, t:t1, c:c1})
    dxt_ev = dxt.subs({a:a1, b:b1, t:t1, c:c1})

    # ARRAY
    M_row = sympy.Matrix([[dxa, dxb]])   #ROW
    M_col = sympy.Matrix([dxa,dxb])        #COLUMN
    M1 = M_row.subs({a:a1, b:b1, t:t1, c:c1})
    M2 = M_col.subs({a:a1, b:b1, t:t1, c:c1})

    # TO NUMPY
    M3 = sympy.matrix2numpy(M1) #riga
    M4 = sympy.matrix2numpy(M2) #colonna

    # ROW-COLUMN PRODUCT
    M5 = np.dot(M3,pcov_lin1)
    M6 = np.dot(M5,M4)
    dz1 = math.sqrt(M6)
    print(f"dz1 = {dz1}")

    M7 = M6 + ((dxc_ev*err_c21)**2) + ((dxt_ev**2)*(pcov1[0,0]))
    dz2 = math.sqrt(M7)
    print(f"dz2 = {dz2}")

    return dz2, x_thr_ev, x_thr2_ev



def th_disp_fourth_woc(a1, b1, t1, pcov_lin1, pcov1, th1, dth1):
    th1 = float(th1)

    a, b, th, t = sympy.symbols('a, b, th, t')
    x_thr =  (t/2) - (b/(2*a)) + sympy.sqrt((t/2 + (b/(2*a)))**2 + ((th-t)*(a*th+b))/a)
    x_thr2 =  (t/2) - (b/(2*a)) - sympy.sqrt((t/2 + (b/(2*a)))**2 + ((th-t)*(a*th+b))/a)
    x_thr_ev = x_thr.subs({a:a1, b:b1, t:t1, th:th1})
    x_thr2_ev = x_thr2.subs({a:a1, b:b1, t:t1, th:th1})


    dxa = sympy.diff(x_thr, a)
    dxb = sympy.diff(x_thr, b)
    dxth = sympy.diff(x_thr, th)
    dxt = sympy.diff(x_thr, t)

    dxth_ev = dxth.subs({a:a1, b:b1, t:t1, th:th1})
    dxt_ev = dxt.subs({a:a1, b:b1, t:t1, th:th1})

    # ARRAY
    M_row = sympy.Matrix([[dxa, dxb]])   #ROW
    M_col = sympy.Matrix([dxa,dxb])        #COLUMN
    M1 = M_row.subs({a:a1, b:b1, t:t1, th:th1})
    M2 = M_col.subs({a:a1, b:b1, t:t1, th:th1})

    # TO NUMPY
    M3 = sympy.matrix2numpy(M1) #riga
    M4 = sympy.matrix2numpy(M2) #colonna

    # ROW-COLUMN PRODUCT
    M5 = np.dot(M3,pcov_lin1)
    M6 = np.dot(M5,M4)
    dz1 = math.sqrt(M6)
    print(f"dz1 = {dz1}")

    M7 = M6 + ((dxth_ev*dth1)**2) + ((dxt_ev**2)*(pcov1[0,0]))
    dz2 = math.sqrt(M7)
    print(f"dz2 = {dz2}")

    return dz2, x_thr_ev, x_thr2_ev



#########################################################################
# ANALYSIS ON TOT VALUE AND DISPERSION FROM CALIBRATION CURVE

def dx_ev1(a1, b1, c1, t1, pcov1):
    y, x, a, b, c, t = sympy.symbols('y, x, a, b, c, t')
    x1=23.38

    # CHARGE FUNCTION
    y = (t/2)-(b/(2*a))+(x/(2*a)) + sympy.sqrt(((t/2)+(b/(2*a))-(x/(2*a)))**2 + (c/a))
    # PARTIAL DERIVATIVES
    dya = sympy.diff(y, a)
    dyb = sympy.diff(y, b)
    dyc = sympy.diff(y, c)
    dyt = sympy.diff(y, t)
    dyx = sympy.diff(y, x)
    dyx_ev = dyx.subs({x:x1, a:a1, b:b1, t:t1, c:c1})
    dyx_err =1.41
    y_ev = y.subs({x:x1, a:a1, b:b1, t:t1, c:c1})

    # ARRAY
    M_row = sympy.Matrix([[dya, dyb, dyc, dyt]])   #ROW
    M_col = sympy.Matrix([dya,dyb,dyc,dyt])        #COLUMN
    M1 = M_row.subs({x:x1, a:a1, b:b1, t:t1, c:c1})
    M2 = M_col.subs({x:x1, a:a1, b:b1, t:t1, c:c1})

    # TO NUMPY
    M3 = sympy.matrix2numpy(M1) #riga
    M4 = sympy.matrix2numpy(M2) #colonna

    # ROW-COLUMN PRODUCT
    M5 = np.dot(M3,pcov1)
    M6 = np.dot(M5,M4)
    dz1 = math.sqrt(M6)
    print(f"dz1 = {dz1}")

    M7 = M6 + ((dyx_ev*dyx_err)**2)
    dz2 = math.sqrt(M7)
    print(f"dz2 = {dz2}")

    return dz2, y_ev


def dtot1(a1, b1, c1, t1, pcov1, q_error):
    y, x, a, b, c, t = sympy.symbols('y, x, a, b, c, t')

    # TOT FUNCTION
    y = a*x + b - (c/(x-t))

    # PARTIAL DERIVATIVES
    dya = sympy.diff(y, a)
    dyb = sympy.diff(y, b)
    dyc = sympy.diff(y, c)
    dyt = sympy.diff(y, t)

    dyx = sympy.diff(y, x)
    dyx_ev = dyx.subs({x:160, a:a1, b:b1, t:t1, c:c1})

    # ARRAY
    M_row = sympy.Matrix([[dya, dyb, dyc, dyt]])   #ROW
    M_col = sympy.Matrix([dya,dyb,dyc,dyt])        #COLUMN
    M1 = M_row.subs({x:160, a:a1, b:b1, t:t1, c:c1})
    M2 = M_col.subs({x:160, a:a1, b:b1, t:t1, c:c1})

    # TO NUMPY
    M3 = sympy.matrix2numpy(M1) #riga
    M4 = sympy.matrix2numpy(M2) #colonna

    # ROW-COLUMN PRODUCT
    M5 = np.dot(M3,pcov1)
    M6 = np.dot(M5,M4)
    dz1 = math.sqrt(M6)
    print(f"dz1 = {dz1}")

    M7 = M6 + ((dyx_ev*q_error)**2)
    dz2 = math.sqrt(M7)
    print(f"dz2 = {dz2}")


    # da = (dya**2)*pcov1[0,0]
    # db = (dyb**2)*pcov1[1,1]
    # dc = (dyc**2)*pcov1[2,2]
    # dt = (dyt**2)*pcov1[3,3]
    #
    # dab = 2*dya*dyb*pcov1[0,1]
    # dac = 2*dya*dyc*pcov1[0,2]
    # dat = 2*dya*dyt*pcov1[0,3]
    # dbc = 2*dyb*dyc*pcov1[1,2]
    # dbt = 2*dyb*dyt*pcov1[1,3]
    # dct = 2*dyc*dyt*pcov1[2,3]
    #
    # dy2 = da + db + dc + dt + dab + dac + dat + dbc + dbt + dct
    # print(dy2)
    # dy = sympy.sqrt(dy2)

    z = y.subs({x:160, a:a1, b:b1, t:t1, c:c1})
    # dz = dy.subs({x:160, a:a1, b:b1, t:t1, c:c1})


    return z, dz2



def dx_ev_wo_c(a1, b1, t1, pcov1, d_th, th1):
    y, x, a, b, t, th = sympy.symbols('y, x, a, b, t, th')
    x1=23.38
    th1=float(th1)

    # CHARGE FUNCTION
    y = (t/2)-(b/(2*a))+(x/(2*a)) + sympy.sqrt(((t/2)+(b/(2*a))-(x/(2*a)))**2 + (((th-t)*(a*th + b))/a))
    # PARTIAL DERIVATIVES
    dya = sympy.diff(y, a)
    dyb = sympy.diff(y, b)
    dyth = sympy.diff(y, th)
    dyt = sympy.diff(y, t)
    dyx = sympy.diff(y, x)
    dyx_ev = dyx.subs({x:x1, a:a1, b:b1, t:t1, th:th1})
    dyx_err =1.41
    dyth_ev = dyx.subs({x:x1, a:a1, b:b1, t:t1, th:th1})
    y_ev = y.subs({x:x1, a:a1, b:b1, t:t1, th:th1})

    # ARRAY
    M_row = sympy.Matrix([[dya, dyb, dyt]])   #ROW
    M_col = sympy.Matrix([dya,dyb,dyt])        #COLUMN
    M1 = M_row.subs({x:x1, a:a1, b:b1, t:t1, th:th1})
    M2 = M_col.subs({x:x1, a:a1, b:b1, t:t1, th:th1})

    # TO NUMPY
    M3 = sympy.matrix2numpy(M1) #riga
    M4 = sympy.matrix2numpy(M2) #colonna

    # ROW-COLUMN PRODUCT
    M5 = np.dot(M3,pcov1)
    M6 = np.dot(M5,M4)
    dz1 = math.sqrt(M6)
    print(f"dz1 = {dz1}")

    M7 = M6 + ((dyx_ev*dyx_err)**2) + ((dyth_ev*d_th)**2)
    dz2 = math.sqrt(M7)
    print(f"dz2 = {dz2}")

    return dz2, y_ev




def dtot2_wo_c(a1, b1, t1, th1, d_th1, pcov1, q_error):
    x, a, b, t, th = sympy.symbols('x, a, b, t, th')
    th1=float(th1)

    y = a*x + b - (((th-t)*(a*th + b))/(x-t))

    dya = sympy.diff(y, a)
    dyb = sympy.diff(y, b)
    dyth = sympy.diff(y, th)
    dyt = sympy.diff(y, t)

    dyx = sympy.diff(y, x)
    dyx_ev = dyx.subs({x:160, a:a1, b:b1, t:t1, th:th1})
    dyth_ev = dyx.subs({x:160, a:a1, b:b1, t:t1, th:th1})


    # ARRAY
    M_row = sympy.Matrix([[dya, dyb, dyt]])   #ROW
    M_col = sympy.Matrix([dya,dyb,dyt])        #COLUMN
    M1 = M_row.subs({x:160, a:a1, b:b1, t:t1, th:th1})
    M2 = M_col.subs({x:160, a:a1, b:b1, t:t1, th:th1})

    # TO NUMPY
    M3 = sympy.matrix2numpy(M1) #riga
    M4 = sympy.matrix2numpy(M2) #colonna

    # ROW-COLUMN PRODUCT
    M5 = np.dot(M3,pcov1)
    M6 = np.dot(M5,M4)
    dz1 = math.sqrt(M6)
    print(f"dz1 = {dz1}")

    M7 = M6 + ((dyx_ev*q_error)**2) + ((dyth_ev*d_th1)**2)
    dz2 = math.sqrt(M7)
    print(f"dz2 = {dz2}")


    # da = (dya**2)*pcov1[0,0]
    # db = (dyb**2)*pcov1[1,1]
    # dt = (dyt**2)*pcov1[2,2]
    # dth = (dyth*d_th1)**2
    #
    # dab = 2*dya*dyb*pcov1[0,1]
    # dat = 2*dya*dyt*pcov1[0,2]
    # dbt = 2*dyb*dyt*pcov1[1,2]
    #
    # dy2 = da + db + dth + dt + dab + dat + dbt + ((dyx_ev*q_error)**2)
    # dy = sympy.sqrt(dy2)

    z = y.subs({x:160, a:a1, b:b1, t:t1, th:th1})
    #dz = dy.subs({x:160, a:a1, b:b1, t:t1, th:th1})

    return z, dz2


def dx_ev_w_c(a1, b1, t1, c1, dc1, pcov1):
    y, x, a, b, c, t = sympy.symbols('y, x, a, b, c, t')
    x1=23.38
    c1=float(c1)

    # CHARGE FUNCTION
    y = (t/2)-(b/(2*a))+(x/(2*a)) + sympy.sqrt(((t/2)+(b/(2*a))-(x/(2*a)))**2 + (c/a))
    # PARTIAL DERIVATIVES
    dya = sympy.diff(y, a)
    dyb = sympy.diff(y, b)
    dyc = sympy.diff(y, c)
    dyt = sympy.diff(y, t)
    dyx = sympy.diff(y, x)
    dyx_ev = dyx.subs({x:x1, a:a1, b:b1, t:t1, c:c1})
    dyx_err =1.41
    dyc_ev = dyc.subs({x:x1, a:a1, b:b1, t:t1, c:c1})

    # ARRAY
    M_row = sympy.Matrix([[dya, dyb, dyt]])   #ROW
    M_col = sympy.Matrix([dya,dyb,dyt])        #COLUMN
    M1 = M_row.subs({x:x1, a:a1, b:b1, t:t1, c:c1})
    M2 = M_col.subs({x:x1, a:a1, b:b1, t:t1, c:c1})

    # TO NUMPY
    M3 = sympy.matrix2numpy(M1) #riga
    M4 = sympy.matrix2numpy(M2) #colonna

    # ROW-COLUMN PRODUCT
    M5 = np.dot(M3,pcov1)
    M6 = np.dot(M5,M4)
    dz1 = math.sqrt(M6)
    print(f"dz1 = {dz1}")

    M7 = M6 + ((dyx_ev*dyx_err)**2) + ((dyc_ev*dc1)**2)
    dz2 = math.sqrt(M7)
    print(f"dz2 = {dz2}")

    y_ev = y.subs({x:x1, a:a1, b:b1, t:t1, c:c1})

    return dz2, y_ev

def dtot2_w_c(a1, b1, t1, c1, dc1, pcov1, q_error):
    x, a, b, t, c = sympy.symbols('x, a, b, t, c')
    c1=float(c1)

    y = a*x + b - (c/(x-t))

    dya = sympy.diff(y, a)
    dyb = sympy.diff(y, b)
    dyc = sympy.diff(y, c)
    dyt = sympy.diff(y, t)

    dyx = sympy.diff(y, x)
    dyx_ev = dyx.subs({x:160, a:a1, b:b1, t:t1, c:c1})
    dyc_ev = dyx.subs({x:160, a:a1, b:b1, t:t1, c:c1})


    # ARRAY
    M_row = sympy.Matrix([[dya, dyb, dyt]])   #ROW
    M_col = sympy.Matrix([dya,dyb,dyt])        #COLUMN
    M1 = M_row.subs({x:160, a:a1, b:b1, t:t1, c:c1})
    M2 = M_col.subs({x:160, a:a1, b:b1, t:t1, c:c1})

    # TO NUMPY
    M3 = sympy.matrix2numpy(M1) #riga
    M4 = sympy.matrix2numpy(M2) #colonna

    # ROW-COLUMN PRODUCT
    M5 = np.dot(M3,pcov1)
    M6 = np.dot(M5,M4)
    dz1 = math.sqrt(M6)
    print(f"dz1 = {dz1}")

    M7 = M6 + ((dyx_ev*q_error)**2) + ((dyc_ev*dc1)**2)
    dz2 = math.sqrt(M7)
    print(f"dz2 = {dz2}")


    # da = (dya**2)*pcov1[0,0]
    # db = (dyb**2)*pcov1[1,1]
    # dt = (dyt**2)*pcov1[2,2]
    # dth = (dyth**2)*(d_th1**2)
    #
    # dab = 2*dya*dyb*pcov1[0,1]
    # dat = 2*dya*dyt*pcov1[0,2]
    # dbt = 2*dyb*dyt*pcov1[1,2]
    #
    # dy2 = da + db + dth + dt + dab + dat + dbt + ((dyx_ev*q_error)**2) + ((dyth_ev*d_th1)**2)
    # dy = sympy.sqrt(dy2)

    z = y.subs({x:160, a:a1, b:b1, t:t1, c:c1})
    # dz = dy.subs({x:160, a:a1, b:b1, t:t1, th:th1})

    return z, dz2
