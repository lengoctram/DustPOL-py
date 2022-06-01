from plt_setup import *
from read import *
from common import *
import align
# ------------------------------------------------------
# [DESCRIPTION] disrupted grain size
# 1. wRAT : angular velocity of grain rotation by radiation
#    input:
#	U = radiation field strength
#	a = grain size
#	lambA = mean wavelength of radiation field
#    output:
#	w_rat = angular velocity of grain rotation
#
# 2. a_disrupt : disrupted grain size
#    input : 
#	UINDEX = U = radiation field strength
#	Smax = tensile strength
#    output : 
#	amax = disrupted grain size#		
# ------------------------------------------------------
def wRAT(U,a,n_gas):
    na = len(a)
    w_rat = np.zeros((len(a)))
    
    for ia in range(na):
        # --------------- Qgam ---------------
        if a[ia] <= lambA/1.8:
            Qgam = 2 * (lambA/a[ia])**(-2.7)
        else:
            Qgam = 0.4

        # --------------- Torq ---------------
        torq_rat = a[ia]**2 *gamma*(U*u_ISRF) * lambA * Qgam / 2

        # --------------- Wrat ---------------
        #FIR = 9.1e-6 * U**(2/3.) * (30/n_gas) * sqrt(100/T_gas) / a[ia]
        FIR = 4.0e-6 * U**(2/3.) * (30./n_gas) * sqrt(100./T_gas) / a[ia]
        t_gas = 8.74e9 * a[ia] * (rho/3.) * (30./n_gas) * sqrt(100./T_gas)
        t_damp = t_gas / (1+FIR)
        I = 8.*pi*rho*a[ia]**5 / 15.
        w_rat[ia] = torq_rat * t_damp / I *(365*24*60*60) # [Hz] : years -> seconds
    
    return w_rat

def a_disrupt(UINDEX,a,n_gas):
    import matplotlib.pyplot as plt   
    wd = 2./a * sqrt(Smax/rho)
    wr = wRAT(UINDEX,a,n_gas)
    #print('check:',shape(a),shape(wd),shape(wr))
    #plt.loglog(a,wd,'-')
    #plt.loglog(a,wr,'--')
    #plt.show()
    #exit(0)
    try:
        acrit=align.intersection(a,array(wd),a,array(wr))[0][0]
        log.info('   *** Checking disruption: \033[1;36m occured \033[0m')
    except:
        lmax = max(where(a<=amax+0.1*amax)[0])
        acrit = a[lmax]
        log.info('   *** Checking disruption: \033[1;36m no \033[0m')
    return acrit