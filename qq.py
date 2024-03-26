from plt_setup import *
from read import *
from scipy import interpolate
# from common import *
import importlib
import common; importlib.reload(common)
from common import alpha

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

def Qext_grain(data,w_new,a_new):

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
    else:                   #oblate
        Qpol= (Qabs2 - Qabs1)
        Qpol_abs = ((Qabs2 + Qsca2) - (Qabs1 +Qsca1))
    
    f_ext = interpolate.RectBivariateSpline(w,a, Qext, kx = 5, ky = 5 )
    f_abs = interpolate.RectBivariateSpline(w,a, Qabs, kx = 5, ky = 5 )
    f_pol_emis = interpolate.RectBivariateSpline(w,a, Qpol, kx = 5, ky = 5 )
    f_pol_abs  = interpolate.RectBivariateSpline(w,a, Qpol_abs, kx = 5, ky = 5 )

    Qext= f_ext(w_new, a_new)
    Qabs= f_abs(w_new, a_new)
    Qpol= f_pol_emis(w_new, a_new)
    Qpol_abs= f_pol_abs(w_new, a_new)

    return [Qext, Qabs, Qpol, Qpol_abs]
