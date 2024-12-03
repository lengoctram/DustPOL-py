from numpy import *
from .read import *
import scipy.integrate as integrate
import warnings
#from common import path
from joblib import Parallel, delayed#, Memory

#suppress warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# Physical constants
# -------------------------------------------------------------------------
H   = 6.625e-27
C   = 3e10
K   = 1.38e-16
amu = 1.6605402e-24 #atomic mass unit
yr  = 365.*24.*60.*60.
# -------------------------------------------------------------------------
# Dust :: w (wavelength), a (grain size), T_dust (dust temperature), alpha (axial ratio)
# -------------------------------------------------------------------------
#alpha = 0.3333
# min/maximum grain size
#amin = arange[0]#1. *1.e-7 # [cm]
#amax = arange[-1]#1. *1.e-4 # [cm]
# w
def wave(path):
    filename = path+"data/LAMBDA.DAT"
    q = genfromtxt(filename,skip_header = 4, dtype=['float'],names=['wave'], usecols= (0))
    w = q['wave'] *1e-4       # in cm
    return w
    
# a
def a_dust(path,UINDEX):
    q = readDC(path+"data/U={:.2f}/SDIST.RES".format(UINDEX),7,1,2,70,2)
    a_gra = q[0,:,0]
    a_sil = q[0,:,1]
    
    return a_gra, a_sil

# T_dust
def T_dust(path,na,UINDEX):
    nT = 200 ##good for prolate shape
    # nT = 125 ##check for oblate shape --> don't work
    T_gra = zeros([na,nT]);
    T_sil = zeros([na,nT]);
    
    dP_dlnT_gra = zeros([na, nT])
    dP_dlnT_sil = zeros([na, nT])
    
    q = genfromtxt(path+"data/U={:.2f}/TEMP.RES".format(UINDEX),skip_header = 8,dtype=['float','float','float','float','float'],names=['T','dP_dlnT','C','U', 'dP/dU'],usecols= (0,1,2,3,4))
    ##print('T=',q['T'])
    for i in range(na):
        for j in range(nT):
            ij = i*nT +j
            ik = na*nT + ij
            T_gra[i,j] = q['T'][ij]
            T_sil[i,j] = q['T'][ik]
            dP_dlnT_gra[i,j] = q['dP_dlnT'][ij]
            dP_dlnT_sil[i,j] = q['dP_dlnT'][ik]
            if T_gra[i,j] <=2.7:
                T_gra[i,j] = 2.7
            if T_sil[i,j] <= 2.7:
                T_sil[i,j] = 2.7

    return [T_gra, T_sil, dP_dlnT_gra, dP_dlnT_sil]

# -------------------------------------------------------------------------
# PLANCK FUNCTION
# -------------------------------------------------------------------------
def planck_1(w,na,T,dP_dlnT):
    nw = len(w)
    nT = len(T[0,:])
    B = zeros([na, nw])

    #printProgressBar(0, na, prefix = '  -> Progress:', suffix = 'Complete', length = 30)
    for i in range(na):
        B_T= zeros([nT, nw])
        for j in range(nT):
            for k in range(nw):
                B_T[j,k] = 2*H*C**2/(w[k])**5 /(exp(H*C/w[k]/K/T[i,j])-1)

        for m in range(nw):
            B[i,m] = integrate.trapz(dP_dlnT[i]*B_T[:,m]/T[i],T[i])

        #printProgressBar(i+1, na, prefix = '  -> Progress:', suffix = 'Complete', length = 30)
    return B

def planck(w,na,T,dP_dlnT):
    nw = len(w)
    nT = len(T[0,:])
    B = zeros([na, nw])

    #printProgressBar(0, na, prefix = '  -> Progress:', suffix = 'Complete', length = 30)
    for i in range(na):
        func_dPdT = dP_dlnT[i]
        func_T    = T[i]
        for k in range(nw):
            #for j in range(nT):
            #    B_T[j,k] = 2*H*C**2/(w[k])**5 /(exp(H*C/w[k]/K/T[i,j])-1)
            B_T      = 2*H*C**2/(w[k])**5 /(exp(H*C/w[k]/K/T[i,:])-1)
            B[i,k] = integrate.trapz(func_dPdT*B_T/func_T, func_T) 
        # for m in range(nw):
        #     B[i,m] = integrate.trapz(dP_dlnT[i]*B_T[:,m]/T[i],T[i])

        #printProgressBar(i+1, na, prefix = '  -> Progress:', suffix = 'Complete', length = 30)
    return B

def planck_equi(w,na,T):
    """Planck function is computed from the equipartition (Drain 2011)
        T = T0*(a/1e-5)^{-1/15} U^{1/6}
    """
    nw = len(w)
    B = zeros([na, nw])

    #printProgressBar(0, na, prefix = '  -> Progress:', suffix = 'Complete', length = 30)
    for i in range(na):
        # for k in range(nw):
        #     B[i,k]      = 2*H*C**2/(w[k])**5 /(exp(H*C/w[k]/K/T[i])-1)

        # for k in range(nw):
        B[i,:]      = 2*H*C**2/(w)**5 /(exp(H*C/w/K/T[i])-1)
    return B

def planck_test(w,na,T,dP_dlnT):
    nw = len(w)
    nT = len(T[0,:])
    B = zeros([na, nw])
    B_f = zeros([na, nw])

    #printProgressBar(0, na, prefix = '  -> Progress:', suffix = 'Complete', length = 30)
    def func_integrate(i,k):
        func_dPdT = dP_dlnT[i]
        func_T    = T[i]
        B_T      = 2*H*C**2/(w[k])**5 /(exp(H*C/w[k]/K/T[i,:])-1)
        B[i,k] = integrate.trapz(func_dPdT*B_T/func_T, func_T) 
        return B

    out=Parallel(n_jobs=-1,verbose=0)(delayed(func_integrate)(i, k) for i in range(na) for k in range(nw))
    ii=0
    for i in range(na):
        for k in range(nw):
            B_f[i,k]=out[ii][i,k]
            ii+=1
    return B_f


def uISRF(dpc):
    C   = 3e10
    q      = readD('./data/Mathis83/GMC_'+str(dpc)+'_ISRF.dat',2,2)
    lamb   = q[0,:] * 1.e-4 #[cm]
    uISRF  = q[1,:] / lamb / C #u_rad

    lmax   = where(abs(lamb-20.e-4) == min(abs(lamb-20.e-4)))
    lrange = lamb[0:lmax[0][0]]
    urange = uISRF[0:lmax[0][0]]
    u_ISRF = trapz(urange,x=lrange)
    return u_ISRF

# compute radiation strength from given radiation field
def LambU(Av,uISRF,dpc):
    C   = 3e10
    Wmax = 20.e-4 # select wavelengths less than 20.e-4 cm
    
    q = readD('./data/Mathis83/GMC_'+str(dpc)+'_Av'+str(Av)+'.dat',2,2)
    lamb = q[0,:] * 1.e-4 #[cm]
    urad = q[1,:] / lamb / C #u_rad
    
    # averaged wavelength in a range of 0.1 - 20 microns
    lmax = where(abs(lamb-Wmax) == min(abs(lamb-Wmax)))
    lrange = lamb[0:lmax[0][0]]
    urange = urad[0:lmax[0][0]]
    
    u_rad = trapz(urange,x=lrange)
    lambA = trapz(urange*lrange,x=lrange)/u_rad
    print('u_rad=',u_rad,'u_ISRF=',uISRF)
    # radiation factor
    U = u_rad / uISRF
    
    return U, lambA
