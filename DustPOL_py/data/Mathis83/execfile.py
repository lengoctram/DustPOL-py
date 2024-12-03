import numpy as np
from numpy import *
import scipy.integrate as integrate
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import simps
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from itertools import cycle

# ============ PLOT ============
# configuration of plot
line = ['-','--','-.']
linecycler = cycle(line)
symb = ['o','^','s','D']
symbcycler = cycle(symb)

font = {'family': 'sans-serif',
    'color':  'black',
        'style': 'normal',
        'weight': 'light',
        'size': 14,
}

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
