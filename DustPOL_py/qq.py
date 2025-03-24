from .plt_setup import *
from .read import *
from scipy import interpolate
# from common import *
# import importlib
# import common; importlib.reload(common)
# from common import alpha
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
# ------------------------------------------------------
# [DESCRIPTION] Qabs and Qsca 
# input :
#	data = file name
#	w_new = applicable wavelength
#	a_new = applicable grain size
# output :
#	Qext = extinction efficiency
#	Qabs = absroption efficiency
#	Qpol = polarization efficiency
# ------------------------------------------------------

def Qext_grain(data,w_new,a_new,alpha=0.3333,fixed=False,wmin=0,wmax=0,dtype='sil'):
    #To calculate the scattering, absorption and polarization efficiencies/cross-sections 
    #given the values in two directions (E||a) and (E|_a)
    w = data[1,:,0]*1e-4 # wavelength in cm
    a = data[0,0,:]*1e-4 # grain size in cm
    Qabs1 = data[4,:,:]
    Qsca1 = data[5,:,:]
    Qabs2 = data[6,:,:]
    Qsca2 = data[7,:,:]

    Qext= (2.*Qabs1 +Qabs2)/3. +(2.*Qsca1 +Qsca2)/3.
    Qabs= (2.*Qabs1 +Qabs2)/3.

    if alpha < 1.0:   #prolate
        Qpol= (Qabs2 - Qabs1)/2.
        Qpol_abs = ((Qabs2 + Qsca2) - (Qabs1 +Qsca1))/2.
    else:             #oblate
        Qpol= Qabs1 - Qabs2#Qabs2 - Qabs1 #-- Hyeseung
        Qpol_abs = ((Qabs2 + Qsca2) - (Qabs1 +Qsca1))
        # Qpol_abs = 1./2*(Qabs2 - Qabs1)

    if (fixed):
        if dtype=='sil':
            Qpol     = Q_fixing(Qpol,a,w,wmin,wmax)
        elif dtype=='car':
            Qext     = Q_fixing(Qext,a,w,wmin,wmax)
            Qpol     = Q_fixing(Qpol,a,w,wmin,wmax)
            Qabs     = Q_fixing(Qabs,a,w,wmin,wmax)
            Qpol_abs = Q_fixing(Qpol_abs,a,w,wmin,wmax)
        else:
            log.error('No dust-type is found!')
        # Qext     = Q_fixing(Qext,a,w,172e-4,300e-4)
        # Qpol     = Q_fixing(Qpol,a,w,172e-4,300e-4)
        # Qabs     = Q_fixing(Qabs,a,w,172e-4,300e-4)
        # Qpol_abs = Q_fixing(Qpol_abs,a,w,172e-4,300e-4)

    f_ext = interpolate.RectBivariateSpline(w,a, Qext, kx = 5, ky = 5 )
    f_abs = interpolate.RectBivariateSpline(w,a, Qabs, kx = 5, ky = 5 )
    f_pol_emis = interpolate.RectBivariateSpline(w,a, Qpol, kx = 5, ky = 5 )
    f_pol_abs  = interpolate.RectBivariateSpline(w,a, Qpol_abs, kx = 5, ky = 5 )

    Qext= f_ext(w_new, a_new)
    Qabs= f_abs(w_new, a_new)
    Qpol= f_pol_emis(w_new, a_new)
    Qpol_abs= f_pol_abs(w_new, a_new)
    return [Qext, Qabs, Qpol, Qpol_abs]

def Qext_grain_astrodust(data,w_new,a_new,alpha=0.3333):
    w = data[1,:,0]*1e-4 # wavelength in cm
    a = data[0,0,:]*1e-4 # grain size in cm
    #Qext = data[2,:,:]
    Qabs1 = data[4,:,:]
    Qsca1 = data[5,:,:]
    Qabs2 = data[6,:,:]
    Qsca2 = data[7,:,:]

    Qext= (2.*Qabs1 +Qabs2)/3. +(2.*Qsca1 +Qsca2)/3.
    Qabs= (2.*Qabs1 +Qabs2)/3.
    Qpol = 1/2*(Qabs1-Qabs2) #thermal dust polarization

    Qext1=Qabs1 +Qsca1
    Qext2=Qabs2 +Qsca2
    Qpol_abs=1./2*(Qext1-Qext2) #extintion polarization

    f_ext = interpolate.RectBivariateSpline(w,a, Qext, kx = 5, ky = 5 )
    f_abs = interpolate.RectBivariateSpline(w,a, Qabs, kx = 5, ky = 5 )
    f_pol_emis = interpolate.RectBivariateSpline(w,a, Qpol, kx = 5, ky = 5 )
    f_pol_abs  = interpolate.RectBivariateSpline(w,a, Qpol_abs, kx = 5, ky = 5 )

    Qext= f_ext(w_new, a_new)
    Qabs= f_abs(w_new, a_new)
    Qpol= f_pol_emis(w_new, a_new)
    Qpol_abs= f_pol_abs(w_new, a_new)
    return [Qext, Qabs, Qpol, Qpol_abs]

def Qext_grain_pah(data,w_new,a_new):
    w = data[1,:,0]*1e-4 # wavelength in cm
    a = data[0,0,:]*1e-4 # grain size in cm
    #Qext = data[2,:,:]
    Qabs  = data[2,:,:]
    Qext  = data[3,:,:]


    f_ext = interpolate.RectBivariateSpline(w,a, Qext, kx = 5, ky = 5 ) ##allowing extra-polation
    f_abs = interpolate.RectBivariateSpline(w,a, Qabs, kx = 5, ky = 5 ) ##allowing extra-polation

    Qext= f_ext(w_new, a_new)
    Qabs= f_abs(w_new, a_new)
    Qpol= np.full_like(w_new,np.nan)
    Qpol_abs= np.full_like(w_new,np.nan)
    return [Qext, Qabs, Qpol, Qpol_abs]

# def Qext_grain_ready(data_file,w_new,a_new,extra=False,wmin=0.0,wmax=0.0):
#     #The scattering, absorption and polarization efficiences/cross-sections are already calculated
#     w_init,a_inti,Qabs_init,Qext_init,Qpol_init = loadtxt(data_file,skiprows=1,unpack=True)

#     #grain size in cm
#     idd = where(w_init==min(w_init))[0]
#     a   = a_inti[idd]*1e-4

#     #wavelength in cm
#     w  = w_init[::len(idd)]*1e-4

#     #Qext
#     Qext = zeros((len(w),len(a)))
#     Qabs = zeros((len(w),len(a)))
#     Qpol = zeros((len(w),len(a)))
#     j=0
#     for i in range(0,len(w)):
#         Qext[i,:] = Qext_init[j:j+len(w)]
#         Qabs[i,:] = Qabs_init[j:j+len(w)]
#         Qpol[i,:] = Qpol_init[j:j+len(w)]
#         j=j+len(w)
    

#     if (extra):
#         ##extrapolate with 1/lambda^{2} toward long wavelengths
#         ## fit a function of 1/lambda**2 for lambda in [10,100] um
#         # w_fit = w[np.where((w>=10e-4) & (w<=100e-4))] #cm
#         # Qpol_fit = Qpol[np.where((w>=10e-4) & (w<=100e-4))]
#         w_fit = w[np.where((w>=wmin) & (w<=wmax))] #cm
#         Qpol_fit = Qpol[np.where((w>=wmin) & (w<=wmax))]

#         def func(x,a):
#             return a/x/x

#         #Qpol_edit = Qpol.copy()
#         w_extra = w[np.where(w>=wmax)]
#         for j in range(0,len(a)):
#             if Qpol_fit[:,j].sum()==0.0:
#                 continue
#             else:
#                 popt=curve_fit(func,w_fit,Qpol_fit[:,j])[0]
#                 # Qpol_edit[:,j][np.where(w>=100e-4)] = func(w_extra,*popt)
#                 Qpol[:,j][np.where(w>=wmax)] = func(w_extra,*popt) 
#     f_ext = interpolate.RectBivariateSpline(w,a, Qext, kx = 5, ky = 5 )
#     f_abs = interpolate.RectBivariateSpline(w,a, Qabs, kx = 5, ky = 5 )
#     f_pol_emis = interpolate.RectBivariateSpline(w,a, Qpol, kx = 5, ky = 5 )
#     # f_pol_abs  = interpolate.RectBivariateSpline(w,a, Qpol_abs, kx = 5, ky = 5 )

#     Qext= f_ext(w_new, a_new)
#     Qabs= f_abs(w_new, a_new)
#     Qpol= f_pol_emis(w_new, a_new)
#     Qpol_abs= 0.0#f_pol_abs(w_new, a_new)
#     return [Qext, Qabs, Qpol, Qpol_abs]

# def Q_fixing(data_input,a,w,wmin,wmax):
#     # w_fit = w[np.where((w>=172e-4) & (w<=300e-4))]
#     # Qpol_fit = Qpol[np.where((w>=172e-4) & (w<=300e-4))]
#     # Qext_fit = Qext[np.where((w>=172e-4) & (w<=300e-4))]
#     w_fit       = w[np.where((w>=wmin) & (w<=wmax))]
#     data_fit    = data_input[np.where((w>=wmin) & (w<=wmax))]
#     def func(x,a):
#         return a/x/x

#     data_out = data_input.copy()
#     w_extra = w[np.where(w>=wmin)]
#     for j in range(0,len(a)):
#         if data_fit[:,j].sum()==0.0:
#             continue
#         else:
#             popt=curve_fit(func,w_fit,data_fit[:,j])[0]
#             data_out[:,j][np.where(w>=wmin)] = func(w_extra,*popt) 
#     return data_out