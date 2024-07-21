import os, numpy
from plt_setup import *
from read import *
import importlib
import rad_func
import scipy.integrate as integrate
import common; importlib.reload(common)
#from common import *
import dnda; importlib.reload(dnda)
from dnda import *
import align; importlib.reload(align)
from align import *
import disrupt; importlib.reload(disrupt)
from disrupt import *
import qq; importlib.reload(qq)
from qq import *
from sys import exit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from astropy import log

from common import cloud, \
                   n_gas, \
                   Urange, \
                   ratd, \
                   working_lam, \
                   gamma, \
                   amin, \
                   amax, \
                   power_index, \
                   f_max, \
                   Smax, \
                   dust_type, \
                   model_layer, \
                   dir_out, \
                   checking, \
                   Urange_tempdist, \
                   GSD_law, \
                   dust_to_gas_ratio, \
                   T_gas, \
                   Tdust, \
                   Data_sil, \
                   Data_mCBE, \
                   parallel, \
                   n_jobs, \
                   bcolors, \
                   options

if model_layer!=1:
        from common import fheat, \
                           Urange2, \
                           fscale, \
                           fscale_car \

numpy.seterr(invalid='ignore')
# ------------------------------------------------------
# [DESCRIPTION]
# This code is used for self-consistent modeling of Pabs 
# and Pem by aligned and disrupted grains due to RATs. 
# ------------------------------------------------------

# ----------- Display some infortant information -------
print('!-----------------------------------------------------!                                     ')
print('!'+f"{bcolors.HEADER}               WELCOME to DustPOL (v1.5)        {bcolors.ENDC}"+'     !')
print('!'+f"{bcolors.blue}              by VARNET Theoretical team         {bcolors.ENDC}"+'    !')
print('!'+f"{bcolors.black}         (1-layer & 2-layer) and (RAT & MRAT)    {bcolors.ENDC}"+'    !')
print('!-----------------------------------------------------!')
print('\n')
# import info_display#; importlib.reload(info_display)
# -------------------------------------------------------------------------
# INFORMATION
# -------------------------------------------------------------------------
print(f"{bcolors.UNDERLINE}{bcolors.green}                      YOUR INPUTs                                   {bcolors.ENDC}")

log.info('\033[1;1;7;36m We are working on a pseudo %d-dust-layer model \033[1;1;7;36m'%(model_layer))
log.info('Cloud       : %s'%(cloud))
log.info('RATD        : %s'%(ratd))
log.info('U           : %s'%(Urange))
log.info('Tdust       : %s'%(Tdust))
log.info('gamma       : %.1f'%(gamma))
log.info('ngas        : %.3e [cm-3]'%(n_gas))
log.info('aligned-dust: %s'%(dust_type))
log.info('amax        : %.3f [um]'%(amax*1e4))
log.info('fmax_RAT    : %.3f'%(f_max))
log.info('GSD         : %s and with power-index: %.1f'%(GSD_law,power_index))
log.info('RAT/MRAT    : %s'%(RATalign))
if RATalign=='MRAT' or RATalign=='mrat':
    log.info('Bfield    : %.3f [muG]'%(Bfield))
    log.info('Ncl      : %.3f'%(Ncl))

print('\n')
print(f"{bcolors.UNDERLINE}{bcolors.green}                        DustPOL                                     {bcolors.ENDC}")    
# ----------- initial condition : wavelength -----------
w = rad_func.wave()
w5000 = where(abs(w-0.5) == min(abs(w-0.5)))
w = w[0:w5000[0][0]]
nw = len(w)
## ------------------------------------------------------
fig1, axis1 = plt.subplots(figsize=(8,6))
fig2, axis2 = plt.subplots(figsize=(8,6))

#for iu in range(0,len(Urange)):
def DustPol(iu):
    #amax=amax*1e-4
    UINDEX = Urange[iu]
    output_file = dir_out+'PemU'+str(UINDEX)+'.dat'
    if (checking):
        # ----------- check for existence ----------------------
        if os.path.exists(output_file):
            log.info('\033[1;7;34m U=%.3f: \033[0m \033[1;5;34m Computed \033[0m '%(UINDEX))
            print   ('   *** Saved at: %s'%(str(dir_out)))
            return
    
    if model_layer==1:
        if not parallel:
            log.info('\033[1;7;32m We are working on U=%.3f \033[0m'%(UINDEX))
    if model_layer==2:
        UINDEX2 = Urange2
        if not parallel:
            log.info('    --> 1-layer: U1=%.3f'%(UINDEX))
            log.info('    --> 2-layer: U2=%.3f'%(UINDEX2))

    # ------- dust grain size -------
    #a = rad_func.a_dust(UINDEX)[1]
    a = rad_func.a_dust(10.0)[1]
    if amax>max(a):
        log.error('SORRY - amax should be %.5f [um] at most [\033[1;5;7;91m failed \033[0m]'%(max(a)*1e4))
        raise IOError('Value of amax!')
    na = len(a)

    # ----- dust temperature and its probability distribution -----
    ##[!warning] UINIDEX here is slightly different from other UINDEX
    ##           Because we pre-calculated tempdist's results
    idx = abs(Urange_tempdist-UINDEX).argmin()
    UINDEX_tempdist = Urange_tempdist[idx]
    qT = rad_func.T_dust(na, UINDEX_tempdist)#T_dust(na,UINDEX)
    T_gra = qT[0]
    T_sil = qT[1]
    dP_dlnT_gra = qT[2]
    dP_dlnT_sil = qT[3]

    if model_layer!=1:
        # -------- Second dust layer --------
        idx2 = abs(Urange_tempdist-UINDEX2).argmin()
        UINDEX2_tempdist = Urange_tempdist[idx2]
        qT2 = rad_func.T_dust(na, UINDEX2_tempdist)#T_dust(na,UINDEX)
        T2_gra = qT2[0]
        T2_sil = qT2[1]
        dP2_dlnT_gra = qT2[2]
        dP2_dlnT_sil = qT2[3]

    # ------- disrupted grain size -------
    lmin = min(where(a>=amin)[0])
    if model_layer==1:
        if (ratd == 'on'):
            a_disr = a_disrupt(UINDEX,a,amax,n_gas)
            if a_disr>amax:
                lmax=abs(log10(a)-log10(amax)).argmin()
            else:
                lmax = abs(log10(a)-log10(a_disr)).argmin()
            # lmax = abs(log10(a)-log10(a_disr)).argmin()
        if (ratd == 'off'):
            lmax = max(where(a<=amax+0.1*amax)[0])
        a = a[lmin:lmax+1]
        na = len(a)
    elif model_layer==2:
        if (ratd == 'on'):
            # -------- First dust layer ----------
            a1_disr = a_disrupt(UINDEX,a,amax,n_gas)
            lmax1 = abs(log10(a)-log10(a1_disr)).argmin()
            # -------- Second dust layer ---------
            a2_disr = a_disrupt(UINDEX2,a,amax,n_gas)
            lmax2 = abs(log10(a)-log10(a2_disr)).argmin()
        if (ratd == 'off'):
            lmax1 = max(where(a<=amax+0.1*amax)[0])
            lmax2 = lmax1
        # -------- First dust layer ----------
        a1 = a[lmin:lmax1+1]
        na1=len(a1)
        # -------- Second dust layer ---------
        a2 = a[lmin:lmax2+1]
        na2=len(a2)

    # ------- grain size distribution -------
    if model_layer==1:
        dn_da_gra = dnda(6,'carbon',a,GSD_law,power_index)
        dn_da_sil = dnda(6,'silicate',a,GSD_law,power_index)
    elif model_layer==2:
        # -------- First dust layer ----------
        dn1_da_gra = dnda(6,'carbon',a1,GSD_law,power_index)
        dn1_da_sil = dnda(6,'silicate',a1,GSD_law,power_index)
        # -------- Second dust layer ---------
        dn2_da_gra = dnda(6,'carbon',a2,GSD_law,power_index)
        dn2_da_sil = dnda(6,'silicate',a2,GSD_law,power_index)

    # ------- alignment -------
    # 1) aligned grain size
    if model_layer==1:
        a_ali  = Aligned_Size(UINDEX,amax,n_gas,T_gas)
        idd_pol= np.where(a>=a_ali)
        if not parallel:
            log.info('   *** amin=%.3e(um), amax=%.2f(um), a_align=%.2f(um)'%(a[0]*1e4,a[-1]*1e4,a_ali*1e4))
            #log.info('   *** len(grain-size)=%i'%(na))
            #log.info('We are working with a_align=%f(um)'%(a_ali*1e4))
            if '{:.5f}'.format(a_ali)>'{:.5f}'.format(a[-1]):
                log.error('\033[1;91m alignment size (a_ali)>maximum grain size \033[0m')
                raise IOError('alignment size')

    elif model_layer==2:
        # -------- First dust layer ---------
        a1_ali  = Aligned_Size(UINDEX,amax,n_gas,T_gas)
        idd1_pol= np.where(a1>=a1_ali)
        # -------- First dust layer ---------
        a2_ali  = Aligned_Size(UINDEX2,amax,n_gas,T_gas)
        idd2_pol= np.where(a2>=a2_ali)
        if not parallel:
            log.info('   *** [First layer]: amin=%.3e(um), amax=%f(um), a1_align=%f(um)'%(a1[0]*1e4,a1[-1]*1e4,a1_ali*1e4))
            log.info('   *** [Second layer]: amin=%.3e(um), amax=%f(um), a2_align=%f(um)'%(a2[0]*1e4,a2[-1]*1e4,a2_ali*1e4))
            
            if '{:.5f}'.format(a1_ali)>'{:.5f}'.format(a1[-1]):
                log.error('\033[1;91m [First layer]: alignment size (a1_ali)>maximum grain size \033[0m')
                raise IOError('alignment size')
            elif '{:.5f}'.format(a2_ali)>'{:.5f}'.format(a2[-1]):
                log.error('\033[1;91m [Second layer]: alignment size (a2_ali)>maximum grain size \033[0m')
                raise IOError('alignment size')


    # 2) alignment function
    if model_layer==1:
        fa = f_ali(UINDEX,a,a_ali,f_max)
    elif model_layer==2:
        # -------- First dust layer ---------
        fa1= f_ali(UINDEX,a1,a1_ali,f_max)
        # -------- Second dust layer --------
        fa2= f_ali(UINDEX2,a2,a2_ali,f_max)
    
    # -------- new dust temperature and its probability distribution --------
    if model_layer==1:
        T_gra = T_gra[lmin:lmax+1, :]
        T_sil = T_sil[lmin:lmax+1, :]
        dP_dlnT_gra = dP_dlnT_gra[lmin:lmax+1, :]
        dP_dlnT_sil = dP_dlnT_sil[lmin:lmax+1, :]
    elif model_layer==2:
        # -------- First dust layer ---------
        T_gra = T_gra[lmin:lmax1+1, :]
        T_sil = T_sil[lmin:lmax1+1, :]
        dP_dlnT_gra = dP_dlnT_gra[lmin:lmax1+1, :]
        dP_dlnT_sil = dP_dlnT_sil[lmin:lmax1+1, :]
        # -------- Second dust layer --------
        T2_gra = T2_gra[lmin:lmax2+1, :]
        T2_sil = T2_sil[lmin:lmax2+1, :]
        dP2_dlnT_gra = dP2_dlnT_gra[lmin:lmax2+1, :]
        dP2_dlnT_sil = dP2_dlnT_sil[lmin:lmax2+1, :]

    # -------- PLANCK FUNCTION -------- 
    if model_layer==1: 
        B_gra = rad_func.planck(w,na,T_gra,dP_dlnT_gra)
        B_sil = rad_func.planck(w,na,T_sil,dP_dlnT_sil)
    elif model_layer==2:
        # -------- First dust layer ---------        
        B_gra = rad_func.planck(w,na1,T_gra,dP_dlnT_gra)
        B_sil = rad_func.planck(w,na1,T_sil,dP_dlnT_sil)
        # -------- Second dust layer --------
        B2_gra = rad_func.planck(w,na2,T2_gra,dP2_dlnT_gra)
        B2_sil = rad_func.planck(w,na2,T2_sil,dP2_dlnT_sil)

    # -------- efficiency factors : Qext, Qpol --------
    if model_layer==1:
        [Qext_sil, Qabs_sil, Qpol_sil, Qpol_abs_sil] = Qext_grain(Data_sil,w,a)        
        [Qext_amCBE, Qabs_amCBE, Qpol_amCBE, Qpol_abs_amCBE] = Qext_grain(Data_mCBE,w,a)
    elif model_layer==2:
        # -------- First dust layer ---------        
        [Qext1_sil, Qabs1_sil, Qpol1_sil, Qpol1_abs_sil] = Qext_grain(Data_sil,w,a1)        
        [Qext1_amCBE, Qabs1_amCBE, Qpol1_amCBE, Qpol1_abs_amCBE] = Qext_grain(Data_mCBE,w,a1)
        # -------- Second dust layer --------        
        [Qext2_sil, Qabs2_sil, Qpol2_sil, Qpol2_abs_sil] = Qext_grain(Data_sil,w,a2)        
        [Qext2_amCBE, Qabs2_amCBE, Qpol2_amCBE, Qpol2_abs_amCBE] = Qext_grain(Data_mCBE,w,a2)

# ==================== Polarized emission ====================
    # -------- Silicate grain --------
    if model_layer==1:
        ##Total intensities
        arr1=dn_da_sil* Qabs_sil* pi* a**2 
        arr2=B_sil.T
        dI_em_sil=np.multiply(arr1,arr2)
        Iem_sil = integrate.simps(dI_em_sil, a)

        ##Polarized intensities
        arr3=dn_da_sil* Qpol_sil* pi* a**2 * fa/2
        dI_pol_sil=np.multiply(arr3,arr2)
        Ipol_sil = integrate.simps(dI_pol_sil, a)

    elif model_layer==2:
        ##Total intensities
        ##1-first layer
        arr1=dn1_da_sil* Qabs1_sil* pi* a1**2 
        arr2=B_sil.T
        dI1_em_sil=np.multiply(arr1,arr2)
        Iem1_sil = integrate.simps(dI1_em_sil, a1)

        ##2-second layer
        arr3=dn2_da_sil* Qabs2_sil* pi* a2**2 
        arr4=B2_sil.T
        dI2_em_sil=np.multiply(arr3,arr4)
        Iem2_sil = integrate.simps(dI2_em_sil, a2)

        ##3-sum over total intensities
        Iem_sil = Iem1_sil + 1.0*fscale*Iem2_sil

        ##Polarized intensities
        ##1-first layer
        arr5=dn1_da_sil[idd1_pol]* Qpol1_sil[:,idd1_pol[0][0]:]* pi* a1[idd1_pol]**2 * fa1[idd1_pol]/2
        arr6=arr2[:,idd1_pol[0][0]:]
        dI1_pol_sil=np.multiply(arr5,arr6)
        Ipol1_sil = integrate.simps(dI1_pol_sil, a1[idd1_pol])

        ##2-second layer
        arr7=dn2_da_sil[idd2_pol]* Qpol2_sil[:,idd2_pol[0][0]:]* pi* a2[idd2_pol]**2 * fa2[idd2_pol]/2
        arr8=B2_sil.T[:,idd2_pol[0][0]:]
        dI2_pol_sil=np.multiply(arr7,arr8)
        Ipol2_sil = integrate.simps(dI2_pol_sil, a2[idd2_pol])

        ##3-sum over polarized intensities
        Ipol_sil = Ipol1_sil + 1.0*fscale*Ipol2_sil

    # -------- Carbonaceous grain --------
    if model_layer==1:
        ##Total intensities
        arr1=dn_da_gra* Qabs_amCBE* pi* a**2
        arr2=B_gra.T
        dI_em_amCBE=np.multiply(arr1,arr2)
        Iem_amCBE  = integrate.simps(dI_em_amCBE, a)

        ##Polarized intensities
        arr3=dn_da_gra* Qpol_amCBE* pi* a**2*fa/2
        dI_pol_amCBE=np.multiply(arr3,arr2)
        Ipol_amCBE = integrate.simps(dI_pol_amCBE, a)

    elif model_layer==2:
        ##Total intensities
        ##1-first layer
        arr1=dn1_da_gra* Qabs1_amCBE* pi* a1**2
        arr2=B_gra.T
        dI1_em_amCBE=np.multiply(arr1,arr2)
        Iem1_amCBE  = integrate.simps(dI1_em_amCBE, a1)

        ##2-second layer
        arr3=dn2_da_gra* Qabs2_amCBE* pi* a2**2
        arr4=B2_gra.T
        dI2_em_amCBE=np.multiply(arr3,arr4)
        Iem2_amCBE  = integrate.simps(dI2_em_amCBE, a2)

        ##3-sum over total intensities        
        Iem_amCBE = Iem1_amCBE + 1.0*fscale*fscale_car*Iem2_amCBE

        ##Polarized intensities
        ##1-first layer
        arr5=dn1_da_gra[idd1_pol]* Qpol1_amCBE[:,idd1_pol[0][0]:]* pi* a1[idd1_pol]**2*fa1[idd1_pol]/2
        arr6=arr2[:,idd1_pol[0][0]:]
        dI1_pol_amCBE=np.multiply(arr5,arr6)
        Ipol1_amCBE = integrate.simps(dI1_pol_amCBE, a1[idd1_pol])

        ##2-second layer
        arr7=dn2_da_gra[idd2_pol]* Qpol2_amCBE[:,idd2_pol[0][0]:]* pi* a2[idd2_pol]**2*fa2[idd2_pol]/2
        arr8 = B2_gra.T[:,idd2_pol[0][0]:]
        dI2_pol_amCBE=np.multiply(arr7,arr8)
        Ipol2_amCBE = integrate.simps(dI2_pol_amCBE, a2[idd2_pol])        

        ##3-sum over polarized intensities
        Ipol_amCBE = 1.0*Ipol1_amCBE + 1.0*fscale*fscale_car*Ipol2_amCBE

    # -------- Total emission --------
    #Total emission intensity, produced by both Sil and Carb
    Iext = (Iem_amCBE +Iem_sil)* w
    
    #If only Sil are aligned, Ipol = Ipol_Sil
    Ipol_emis_sil  = Ipol_sil *w
    ratio_emis_sil = Ipol_emis_sil/Iext *100

    #The cas when only carbonaceous are aligned
    Ipol_emis_car  = Ipol_amCBE*w
    ratio_emis_car = Ipol_emis_car/Iext*100

    #The case when both sil and carbonaceous have been aligned
    Ipol_emis_tot  = (Ipol_sil + Ipol_amCBE)*w
    ratio_emis_tot = Ipol_emis_tot/Iext *100

    # #Test case
    # Itest_tot      = (fscale*Ipol2_sil + fscale*fscale_car*Ipol2_amCBE)*w
    # ratio_test_tot = Itest_tot/Iext *100

    #return w*1e4,ratio_emis_sil,ratio_emis_tot#,Iem1_sil,fscale*Iem2_sil,Iem1_amCBE,fscale*fscale_car*Iem2_amCBE
    #return w*1e4,Iem_sil,Ipol_amCBE,Ipol_sil ##<<<---- good for testing ...
    #return w*1e4, Ipol1_sil*w, (Iem1_sil+Iem1_amCBE)*w, fscale*Ipol2_sil*w, fscale*Iem2_sil*w#Ipol1_amCBE*w, fscale*fscale_car*Ipol2_amCBE*w, Iext,ratio_emis_sil,ratio_emis_tot,ratio_test_tot
    ## ======================================================================
    ## <<<<<====================>>>>> [OUTPUT] <<<<<====================>>>>>
    ## ======================================================================
    ## --------------- Pem ---------------
    #if not os.path.exists(output_file):
    file1 = open(output_file,"w+")
    file1.write('#  wave   Pem_sil \t\t Pem_car \t\t Pem_tot \n')
    for iw in range(0,len(w)):
        file1.write('%9.4f %.4e %.4e %.4e \n'%(w[iw]*1e4, ratio_emis_sil[iw], ratio_emis_car[iw], ratio_emis_tot[iw]))
    file1.close()

    # -------------------------------------------------------------------------
    # write the output information
    # -------------------------------------------------------------------------
    f=open(dir_out+'output.info','w')
    f.write('Date           : %s     '%datetime.datetime.now() +'\n')
    f.write('Cloud          : %s      '%str(cloud) +'\n')
    f.write('GSD            : %s     '%GSD_law +'\n')
    f.write('mean_wavelength: %f (um)'%(lambA*1.e4) +'\n')
    f.write('Tgas           : %f (K) '%T_gas +'\n')
    f.write('gamma          : %f     '%gamma +'\n')
    if GSD_law=='MNR':
        f.write('beta(MRN_power): %f     '%power_index +'\n')
    f.write('dust_gas_ratio : %f     '%dust_to_gas_ratio +'\n')
    f.write('fmax           : %f     '%f_max)
    f.close()
    # plt.semilogx(w*1e4,ratio_emis_tot,'-',color='gray')
    #return w*1e4,ratio_emis_sil,ratio_emis_tot

def plot_check():
    x1 = 7
    x2 = 6.e3
    y1 = -0.5
    y2 = 30.9
    nxax = 20
    nyax = 40

    percentage=[]
    for UINDEX in Urange:
        cstyle = next(clrcycler)
        lstyle = next(lncycler)
        #fig1,axis1 = plt.subplots()           
        #Psmth_em = savgol_filter(ratio_emis_sil,21,3)
        #ratio_emis_sil = Psmth_em
        output_file = dir_out+'PemU'+str(UINDEX)+'.dat'
        if dust_type=='sil':
            w,ratio_emis=loadtxt(output_file,skiprows=1,usecols=(0,1),unpack=True)
        elif dust_type=='car':
            w,ratio_emis=loadtxt(output_file,skiprows=1,usecols=(0,2),unpack=True)
        elif dust_type=='sil+car':
            w,ratio_emis=loadtxt(output_file,skiprows=1,usecols=(0,3),unpack=True)
        axis1.plot(w, ratio_emis, lw = 1.6, c=cstyle, ls=lstyle, label = "U={}".format(UINDEX))
        leg = axis1.legend(loc='upper right', fontsize=15, frameon=False)
        custm_legend(leg)
                    
        xax = logspace(log10(x1),log10(x2),nxax+1)
        yax = linspace(y1,y2,nyax+1)
        # if ratd == 'off':
        #     axis1.text(xax[1],yax[nyax-4],r'aligned dust:%s'%(dust_type),weight="bold")
        #     axis1.text(xax[1],yax[nyax-7],r'no disruption',weight="bold")
        #     axis1.text(xax[1],yax[nyax-10],r'r = %3.1f'%(alpha))
        # else:
        #     axis1.text(xax[1],yax[nyax-4],r'aligned-dust:%s'%(dust_type),weight="bold")
        #     axis1.text(xax[1],yax[nyax-7],r'RATD',weight="bold")
        #     axis1.text(xax[1],yax[nyax-10],r'Smax = 10$^%1d$ erg cm$^{-3}$'%(log10(Smax)))
        #     axis1.text(xax[1],yax[nyax-13],r'r = %3.1f'%(alpha))

        axis1.axis([x1,x2,y1,y2])
        axis1.set_xlabel('wavelength$\\,(\\mu$$m$)')
        axis1.set_ylabel('P$\\,(\\%)$')
        axis1.set_xscale('log')
        custm_axis(axis1,0.5,-0.06,-0.10,0.5)           
        #fig1.savefig(dir_out+'Pem_spec_U'+str(U)+'.pdf')
        # fig1.savefig(dir_out+'Pem_spec_U.pdf')
                
        fsil = interp1d(w,ratio_emis)
        percentage.append(fsil(working_lam))
    axis2.plot(Urange,percentage,'o-',color='black')
    #axis2.axis([x1,x2,y1,y2])
    
    if ratd == 'off':
        axis1.text(xax[1],yax[nyax-4],r'aligned dust:%s'%(dust_type),weight="bold")
        axis1.text(xax[1],yax[nyax-7],r'no disruption',weight="bold")
        axis1.text(xax[1],yax[nyax-10],r'r = %3.1f'%(alpha))

        axis2.text(0.03,0.9,r'dust-type:%s'%(dust_type),weight="bold",va='center',transform=axis2.transAxes)
        axis2.text(0.03,0.8,r'no disruption',weight="bold",va='center',transform=axis2.transAxes)
        axis2.text(0.04,0.7,r'r = %3.1f'%(alpha),va='center',transform=axis2.transAxes)
    else:
        axis1.text(xax[1],yax[nyax-4],r'aligned-dust:%s'%(dust_type),weight="bold")
        axis1.text(xax[1],yax[nyax-7],r'RATD',weight="bold")
        axis1.text(xax[1],yax[nyax-10],r'Smax = 10$^%1d$ erg cm$^{-3}$'%(log10(Smax)))
        axis1.text(xax[1],yax[nyax-13],r'r = %3.1f'%(alpha))

        axis2.text(0.03,0.9,r'dust-type:%s'%(dust_type),weight="bold",va='center',transform=axis2.transAxes)
        axis2.text(0.03,0.8,r'RATD',weight="bold",va='center',transform=axis2.transAxes)
        axis2.text(0.03,0.7,r'Smax = 10$^%1d$ erg cm$^{-3}$'%(log10(Smax)),va='center',transform=axis2.transAxes)
        axis2.text(0.03,0.6,r'r = %3.1f'%(alpha),va='center',transform=axis2.transAxes)
        axis2.text(0.7,0.9,r'$\lambda$ = %3.1f$\, \mu$m'%(working_lam),va='center',transform=axis2.transAxes)
    
    axis2.set_xscale('log')
    axis2.set_xlabel('U')
    axis2.set_ylabel('P$\\,(\\%)$')
    # fig2.savefig(dir_out+'Pem_U'+ str(U)+'.pdf')
    # fig2.savefig(dir_out+'Pem_U.pdf')
    # #plt.show()

    # # os.system('open ' + dir_out+'Pem_spec_U'+str(U)+'.pdf')
    # # os.system('open ' + dir_out+'Pem_U'+ str(U)+'.pdf')
    # os.system('open ' + dir_out+'Pem_spec_U.pdf')
    # os.system('open ' + dir_out+'Pem_U.pdf')
    plt.show()

if __name__=='__main__':
    if (parallel):
        from joblib import Parallel, delayed#, Memory
        out=Parallel(n_jobs=n_jobs,verbose=1)(delayed(DustPol)(i) for i in range(len(Urange)))
    else: 
        for iu in range(0,len(Urange)):
            DustPol(iu)


    if options.show=='Yes' or options.show=='yes':
        plot_check()
