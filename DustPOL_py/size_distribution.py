from .plt_setup import *
from .read import *
# from common import *
# import importlib
# import common; importlib.reload(common)
# from common import mgas,dust_to_gas_ratio
from scipy.special import erf

# ------------------------------------------------------
# [DESCRIPTION] GRAIN SIZE DISTRIBUTION dn/da
# input:
#	INDEX = 0 - 14 at Rv
#	dusttype = 'carbon' for carbonaceous or PAHs, 'silicate' for silicate
#	aef = effective grain size
#	sizedist = 'MRN', 'WD01', 'DL07'
# output:
#	dnda
# ------------------------------------------------------

mgas = 1.3*1.6605402e-24 #atomic mass unit #90%H + 10%He
def AMRN_dust_gas_ratio(a0, a1, rho_dust, ratio=0.01, beta=-3.5):
    f1= ratio*mgas#0.01*mgas
    #f2=8./3*pi*rho_dust*(pow(a1,0.5)-pow(a0,0.5))
    if float(beta)==-4.0:
        f2= 4./3*pi*rho_dust*(log(a1)-log(a0))
    else:
        f2=abs(4./3*pi*rho_dust*(pow(a1,4+beta)-pow(a0,4+beta)) * 1./(4+beta))
    return f1/f2

def AMRN_sil(a0, a1, rho_s,rho_c, ratio=0.01, beta=-3.5):
    f1= ratio*mgas#0.01*mgas
    #f2=8./3*pi*rho_dust*(pow(a1,0.5)-pow(a0,0.5))
    if float(beta)==-4.0:
        f2= 4./3*pi*(log(a1)-log(a0))
    else:
        f2=abs(4./3*pi*(pow(a1,4+beta)-pow(a0,4+beta)) * 1./(4+beta))
    return f1/f2 * 1.12/(1.12*rho_s+rho_c)

def AMRN_car(a0, a1, rho_s, rho_c, ratio=0.01, beta=-3.5):
    f1= ratio*mgas#0.01*mgas
    #f2=8./3*pi*rho_dust*(pow(a1,0.5)-pow(a0,0.5))
    if float(beta)==-4.0:
        f2= 4./3*pi*(log(a1)-log(a0))
    else:
        f2=abs(4./3*pi*(pow(a1,4+beta)-pow(a0,4+beta)) * 1./(4+beta))
    return f1/f2 * 1./(1.12*rho_s+rho_c)


def dnda_astro(acm,sizedist='MRN',MRN_params=None):
    #---AstroDust distribution----------------
    ##MRN_params are ultized if and only if sizedist is MRN
    if sizedist=='MRN':
        a0,a1,rho_dust,ratio,beta=MRN_params
        #Normalization constant for a fixed 
        C = AMRN_dust_gas_ratio(a0,a1,rho_dust,ratio=ratio,beta=beta)
        dnda = C*pow(acm,beta) #* np.exp(-acm/a1)
        return dnda       
    else:
        BAd = 3.31e-10 #H-1
        a0Ad=63.8*1e-8 #AA-->cm
        sigAd=0.353
        A0=2.97e-5 #H-1
        A1=-3.40
        A2=-0.807
        A3=0.157
        A4=7.96e-3
        A5=-1.68e-3    

        dnda = BAd/acm * np.exp(-(np.log(acm/a0Ad))**2 /2/sigAd**2) + \
               A0/acm*np.exp( A1*np.log(acm/1e-8) + A2*(np.log(acm/1e-8))**2 + A3*(np.log(acm/1e-8))**3 + A4*(np.log(acm/1e-8))**4 + A5*(np.log(acm/1e-8))**5)
        return dnda

def dnda(INDEX,dusttype,aef,sizedist,power_index,dust_to_gas_ratio=0.01):
    # if (sizedist==GSD_law):
    #     if(sizedist == 'MRN'):
    #         #log.info('   *** [re-check] MRN with the power of %f [\033[1;7;34m ok \033[0m]'%(power_index))
    #     elif (sizedist=='WD01') or (sizedist=='DL07'):
    #         #log.info('   *** [re-check] %s grain-size distribution [\033[1;7;34m ok \033[0m]'%(sizedist))
    #     else:
    #         log.error('   \033[1;91m Grain-size distribution is not valid [MRN/WD01/DL07] \033[0m')
    #         raise IOError('Grain size distribution')
    # else:
    #     #log.info('   *** [re-check] %s grain-size distribution [\033[1;5;7;91m failed \033[0m]'%(sizedist))
    #     log.error('       \033[1;91m Grain-size distribution is not valid [use MRN/WD01/DL07] \033[0m')
    #     raise IOError('Grain size distribution')
    #if dusttype=='silicate':
    #    log.info('   *** [re-check] We are using silicate grains')
    #elif dusttype=='carbon':
    #    log.info('   *** [re-check] We are using carbon grains')
    #else:
    #    log.error('%s is not defined!'%(sizedist))
    
    na = len(aef)
    dnda_gr    = np.zeros(na)
    for i in range(na):
        a = aef[i] ##cm

        #---MRN distribution----------------
        if(sizedist == 'MRN'):
            #log.info('We are using the MRN-like distribution with the power of %f '%(power_index))
            #AMRN=AMRN_dust_gas_ratio(aef[0], aef[-1], rhoo, ratio=dust_to_gas_ratio, beta=MRN_power)
            #dnda    = AMRN* pow(a,MRN_power)
            if dusttype=='silicate':
                #print ('[dnda]: We are using silicate grains')
                AMRN=AMRN_sil(aef[0], aef[-1], 3.0,2.2, ratio=dust_to_gas_ratio, beta=power_index)
            elif dusttype=='carbon':
                #print ('[dnda]: We are using carbon grains')
                AMRN=AMRN_car(aef[0], aef[-1], 3.0,2.2, ratio=dust_to_gas_ratio, beta=power_index)
            else:
                raise IOError('%s is not found!'%dusttype)
            dnda    = AMRN* pow(a,power_index)

        else:
            #log.info('We are using %s grain-size distribution'%(sizedist))
            data = readD('./data/values.dat',2,10)
            if (dusttype == 'carbon'):
                ALPHA = data[0,INDEX]
                BETA  = data[1,INDEX]
                A_T   = data[2,INDEX]*1.E-4
                A_C   = data[3,INDEX]*1.E-4
                C_j   = data[4,INDEX]
                C_MRN = 10**(-25.13)
                rhoo  = 2.2
            if (dusttype == 'silicate'):
                ALPHA = data[5,INDEX]
                BETA  = data[6,INDEX]
                A_T   = data[7,INDEX]*1.E-4
                A_C   = 1.E-5
                C_j   = data[8,INDEX]
                C_MRN = 10**(-25.11)
                rhoo  = 3.0
            #BC5       = data[9,INDEX]
            #if(sizedist == 'DL07'):
            #    BC5   = 0.92*BC5
            B1        = 2.0496E-7
            B2        = 9.6005E-11
    
        
            dnda=(C_j/a)*(a/A_T)**ALPHA #the factor C_j/a in ==. 4
            if (BETA >= 0.):
                dnda = dnda*(1.+BETA*a/A_T)
            else:
                dnda = dnda/(1.-BETA*a/A_T)
            #endif

            if (a > A_T):
                dnda = dnda*np.exp(((A_T-a)/A_C)**3)  ##Silicate grain
            else: 
                dnda = dnda
            #endif

            if (dusttype == 'carbon'):
                dnda_nonlog    = dnda
            
                if(sizedist == 'WD01'):
                    a01 = 3.5E-8
                    a02 = 3.E-7
                    SIG1= 0.4
                    SIG2= 0.4
                    BC5 = data[9,INDEX]
                    dndaVSG=(B1/a)*np.exp(-0.5*(np.log(a/a01)/SIG1)**2)+ (B2/a)*np.exp(-0.5*(np.log(a/a02)/SIG2)**2)
                    if (dndaVSG >=  0.0001*dnda):
                        dnda    = dnda_nonlog+BC5*dndaVSG
                       
                elif(sizedist == 'DL07'):
                    a01    = 4.E-8
                    a02    = 2.E-7
                    SIG1= 0.4
                    SIG2= 0.55
                    BC5   = 0.92*BC5

                    a0        = np.array([4.0,20.0]) #angstrom
                    a0        = a0*(1e-8)    #cm
                    a_min    = 3.556e-8    #cm
                    sigma    = np.array([0.4, 0.55])

                    a_m        = a0*np.exp(3.*sigma**2.)
                    mC        = 12.*(1.67e-24)
                    rho_C    = 2.2

                    bJ        = np.array([BC5*0.75,BC5*0.25])
                    bJ        = bJ*(1e-5)

                    xj        = np.log(a_m/a_min)/(sigma*np.sqrt(2))
                    n0_j    = (3./(2.*np.pi)**1.5)*(np.exp(4.5*sigma**2.)/(1+erf(xj)))*(mC/(rho_C*a_m**3.*sigma))*bJ
                        
                    supl    = (np.log(a/a0))**2./(2.*sigma**2.)

                    #if(dusttype == 'carbon'):
                    dnda = dnda_nonlog + ((n0_j/a)*np.exp(-supl)).sum()
        dnda_gr[i]  = dnda

    return dnda_gr    

def dnda_ref(INDEX,dusttype,aef,sizedist,power_index,dust_to_gas_ratio=0.01):
    # if (sizedist==GSD_law):
    #     if(sizedist == 'MRN'):
    #         #log.info('   *** [re-check] MRN with the power of %f [\033[1;7;34m ok \033[0m]'%(power_index))
    #     elif (sizedist=='WD01') or (sizedist=='DL07'):
    #         #log.info('   *** [re-check] %s grain-size distribution [\033[1;7;34m ok \033[0m]'%(sizedist))
    #     else:
    #         log.error('   \033[1;91m Grain-size distribution is not valid [MRN/WD01/DL07] \033[0m')
    #         raise IOError('Grain size distribution')
    # else:
    #     #log.info('   *** [re-check] %s grain-size distribution [\033[1;5;7;91m failed \033[0m]'%(sizedist))
    #     log.error('       \033[1;91m Grain-size distribution is not valid [use MRN/WD01/DL07] \033[0m')
    #     raise IOError('Grain size distribution')
    #if dusttype=='silicate':
    #    log.info('   *** [re-check] We are using silicate grains')
    #elif dusttype=='carbon':
    #    log.info('   *** [re-check] We are using carbon grains')
    #else:
    #    log.error('%s is not defined!'%(sizedist))
    
    na = len(aef)
    dnda_gr    = np.zeros(na)
    for i in range(na):
        a = aef[i]

        #---MRN distribution----------------
        # if(sizedist == 'MRN'):
            
        #log.info('We are using the MRN-like distribution with the power of %f '%(power_index))
        #AMRN=AMRN_dust_gas_ratio(aef[0], aef[-1], rhoo, ratio=dust_to_gas_ratio, beta=MRN_power)
        #dnda    = AMRN* pow(a,MRN_power)
        if dusttype=='silicate':
            #print ('[dnda]: We are using silicate grains')
            AMRN=AMRN_sil(aef[0], aef[-1], 3.0,2.2, ratio=dust_to_gas_ratio, beta=power_index)
        if dusttype=='carbon':
            #print ('[dnda]: We are using carbon grains')
            AMRN=AMRN_car(aef[0], aef[-1], 3.0,2.2, ratio=dust_to_gas_ratio, beta=power_index)
        dnda_MRN    = AMRN* pow(a,power_index)

        # else:
        #log.info('We are using %s grain-size distribution'%(sizedist))
        data = readD('./data/values.dat',2,10)
        if (dusttype == 'carbon'):
            ALPHA = data[0,INDEX]
            BETA  = data[1,INDEX]
            A_T   = data[2,INDEX]*1.E-4
            A_C   = data[3,INDEX]*1.E-4
            C_j   = data[4,INDEX]
            C_MRN = 10**(-25.13)
            rhoo  = 2.2
        if (dusttype == 'silicate'):
            ALPHA = data[5,INDEX]
            BETA  = data[6,INDEX]
            A_T   = data[7,INDEX]*1.E-4
            A_C   = 1.E-5
            C_j   = data[8,INDEX]
            C_MRN = 10**(-25.11)
            rhoo  = 3.0
        #BC5       = data[9,INDEX]
        #if(sizedist == 'DL07'):
        #    BC5   = 0.92*BC5
        B1        = 2.0496E-7
        B2        = 9.6005E-11
    
        
        dnda=(C_j/a)*(a/A_T)**ALPHA #the factor C_j/a in ==. 4
        if (BETA >= 0.):
            dnda = dnda*(1.+BETA*a/A_T)
        else:
            dnda = dnda/(1.-BETA*a/A_T)
        #endif

        if (a > A_T):
            dnda = dnda*np.exp(((A_T-a)/A_C)**3)  ##Silicate grain
        else: 
            dnda = dnda
        #endif

        if (dusttype == 'carbon'):
            dnda_nonlog    = dnda
            
            if(sizedist == 'WD01'):
                a01 = 3.5E-8
                a02 = 3.E-7
                SIG1= 0.4
                SIG2= 0.4
                BC5 = data[9,INDEX]
                dndaVSG=(B1/a)*np.exp(-0.5*(np.log(a/a01)/SIG1)**2)+ (B2/a)*np.exp(-0.5*(np.log(a/a02)/SIG2)**2)
                if (dndaVSG >=  0.0001*dnda):
                    dnda    = dnda_nonlog+BC5*dndaVSG
                       
            elif(sizedist == 'DL07'):
                a01    = 4.E-8
                a02    = 2.E-7
                SIG1= 0.4
                SIG2= 0.55
                BC5   = 0.92*BC5

                a0        = np.array([4.0,20.0]) #angstrom
                a0        = a0*(1e-8)    #cm
                a_min    = 3.556e-8    #cm
                sigma    = np.array([0.4, 0.55])

                a_m        = a0*np.exp(3.*sigma**2.)
                mC        = 12.*(1.67e-24)
                rho_C    = 2.2

                bJ        = np.array([BC5*0.75,BC5*0.25])
                bJ        = bJ*(1e-5)

                xj        = np.log(a_m/a_min)/(sigma*np.sqrt(2))
                n0_j    = (3./(2.*np.pi)**1.5)*(np.exp(4.5*sigma**2.)/(1+erf(xj)))*(mC/(rho_C*a_m**3.*sigma))*bJ
                        
                supl    = (np.log(a/a0))**2./(2.*sigma**2.)

                #if(dusttype == 'carbon'):
                dnda = dnda_nonlog + ((n0_j/a)*np.exp(-supl)).sum()
        dnda_gr[i]  = dnda + dnda_MRN

    return dnda_gr
