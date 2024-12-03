import numpy as np
from numpy import *

# Data files
path = r'/Users/kasi/Documents/work_dust/data/'
dataR = r'Mathis83/'
dataC = r'Qextpol/'
Rfile = ['GMC_5kpc_ISRF','GMC_5kpc_AV0','GMC_5kpc_AV3','GMC_5kpc_AV5','GMC_5kpc_AV10','GMC_5kpc_AV20','GMC_5kpc_AV50','GMC_10kpc_ISRF.dat']
Cfile = ['Qextpol_wave_size_gra_new.dat','Qextpol_wave_size_sil_new.dat']

c = 3*10**10 #cm/s

# read data : radiation field
def readD(iv):
    file=open(path+dataR+Rfile[iv]+'.dat','r')
    data=[]
    for ln in file:
        temp1 = ln.split()
        temp2 = [n for n in temp1]
        data.append(temp2)
    ns = 2 #beginning line of data
    dataf = np.zeros((2,len(data)-ns))
    for j in range(ns,len(data)):
        dataf[0,j-ns] = data[j][0] #wavelength
        dataf[0,j-ns] = dataf[0,j-ns]*1.e-4 #(cm)
        dataf[1,j-ns] = data[j][1] #lamb*4pi*J
        dataf[1,j-ns] = dataf[1,j-ns]/dataf[0,j-ns]/c #u_rad
    return dataf

# read data : cross-section
def readC(ic):
    file=open(path+dataC+Cfile[ic],'r')
    data=[]
    for ln in file:
        temp1 = ln.split()
        temp2 = [n for n in temp1]
        data.append(temp2)
    ns = 1
    dataf = np.zeros((4,len(data)-ns))
    for j in range(ns,len(data)):
        dataf[0,j-ns] = data[j][0] # wavelength [microns]
#        dataf[0,j-ns] = dataf[0,j-ns] * 1.e-4 # [cm]
        dataf[1,j-ns] = data[j][1] # grain size [microns]
#        dataf[1,j-ns] = dataf[1,j-ns] * 1.e-4 # [cm]
        dataf[2,j-ns] = data[j][2] # Qext
        dataf[3,j-ns] = data[j][3] # Qpol
    return dataf
