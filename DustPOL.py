import os
from plt_setup import *
from read import *
from common import *

from dnda import *
from align import *
from disrupt import *
from qq import *
import rad_func
from sys import exit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import scipy.integrate as integrate
numpy.seterr(invalid='ignore')
# ------------------------------------------------------
# [DESCRIPTION]
# This code is used for self-consistent modeling of Pabs and Pem by aligned and disrupted grains due to RATs. 
# ------------------------------------------------------

# ----------- initial condition : wavelength -----------
w = rad_func.wave()
w5000 = where(abs(w-0.5) == min(abs(w-0.5)))
w = w[0:w5000[0][0]]
nw = len(w)
# ------------------------------------------------------
fig1, axis1 = plt.subplots(figsize=(8,6))
fig2, axis2 = plt.subplots(figsize=(8,6))

#for iu in range(0,len(Urange)):

def DustPol(iu):
    UINDEX = Urange[iu]
    output_file = dir_out+'PemU'+str(UINDEX)+'.dat'
    if (checking):
        # ----------- check for existence ----------------------
        if os.path.exists(output_file):
            log.info('\033[1;7;34m U=%.3f: \033[0m \033[1;5;34m Computed \033[0m '%(UINDEX))
            print   ('   *** Saved at: %s'%(str(dir_out)))
            return
    log.info('\033[1;7;32m We are working on U=%.3f \033[0m'%(UINDEX))

    # ------- dust grain size -------
    #a = rad_func.a_dust(UINDEX)[1]
    a = rad_func.a_dust(10.0)[1]
    if amax>max(a):
        log.error('SORRY - amax should be %.5f [um] at most [\033[1;5;7;91m failed \033[0m]'%(max(a)*1e4))
        raise IOError('Value of amax!')
    na = len(a)

    # ----- dust temperature and its probability distribution -----
    ##[!warning] UINIDEX here is slightly different from other UINDEX
    ##           Because we tabulate the DustEM's results 
    idx = abs(Urange_DustEM-UINDEX).argmin()
    UINDEX_dustem = Urange_DustEM[idx]
    qT = rad_func.T_dust(na,UINDEX_dustem)#T_dust(na,UINDEX)
    T_gra = qT[0]
    T_sil = qT[1]
    dP_dlnT_gra = qT[2]
    dP_dlnT_sil = qT[3]

    # ------- disrupted grain size -------
    lmin = min(where(a>=amin)[0])
    if (ratd == 'on'):
        a_disr = a_disrupt(UINDEX,a,n_gas)
        lmax = abs(log10(a)-log10(a_disr)).argmin()
    if (ratd == 'off'):
        lmax = max(where(a<=amax+0.1*amax)[0])
    a = a[lmin:lmax+1]
    na = len(a)

    # ------- grain size distribution -------
    dn_da_gra = dnda(6,'carbon',a,GSD_law)
    dn_da_sil = dnda(6,'silicate',a,GSD_law)

    # ------- alignment -------
    # 1) aligned grain size
    a_ali = Aligned_Size(UINDEX,n_gas,T_gas)
    log.info('   *** amin=%f(um), amax=%f(um), a_align=%f(um)'%(a[0]*1e4,a[-1]*1e4,a_ali*1e4))
    #log.info('   *** len(grain-size)=%i'%(na))
    #log.info('We are working with a_align=%f(um)'%(a_ali*1e4))
    if '{:.5f}'.format(a_ali)>'{:.5f}'.format(a[-1]):
        log.error('\033[1;91m alignment size (a_ali)>maximum grain size \033[0m')
        raise IOError('alignment size')
    # 2) alignment function
    fa = f_ali(a,a_ali)
    
    # -------- new dust temperature and its probability distribution --------
    T_gra = T_gra[lmin:lmax+1, :]
    T_sil = T_sil[lmin:lmax+1, :]
    dP_dlnT_gra = dP_dlnT_gra[lmin:lmax+1, :]
    dP_dlnT_sil = dP_dlnT_sil[lmin:lmax+1, :]
        
    # -------- PLANCK FUNCTION --------  
    B_gra = rad_func.planck(w,na,T_gra,dP_dlnT_gra)
    B_sil = rad_func.planck(w,na,T_sil,dP_dlnT_sil)

    # -------- efficiency factors : Qext, Qpol --------
    Data = rad_func.readDC('data/Q_aSil2001_'+str(alpha)+'_p20B.DAT',4,4,70,800,8)
    [Qext_sil, Qabs_sil, Qpol_sil, Qpol_abs_sil] = Qext_grain(Data,w,a)
        
    Data = rad_func.readDC('data/Q_amCBE_'+str(alpha)+'.DAT',4,4,100,800,8)
    [Qext_amCBE, Qabs_amCBE, Qpol_amCBE, Qpol_abs_amCBE] = Qext_grain(Data,w,a)

# ==================== Polarized absorption & emission ====================
    Iem_amCBE = np.zeros(nw) # total emission
    Iem_sil = np.zeros(nw)

    Ipol_amCBE = np.zeros(nw)    # polarizatione emission
    Ipol_sil = np.zeros(nw)

    I_pol_abs_amCBE = np.zeros(nw) # observed polarization
    I_pol_abs_sil = np.zeros(nw)
       
    for i in range(nw):
        dI_em_sil = np.zeros([nw, na])
        dI_em_amCBE = np.zeros([nw, na])
    
        dI_pol_sil= np.zeros([nw, na])
        dI_pol_amCBE= np.zeros([nw, na])
    
        dI_pol_abs_sil= np.zeros([nw, na])
        dI_pol_abs_amCBE= np.zeros([nw, na])
        for k in range(na):
            #dI_em_sil[i, k] = dn_da_sil[k]* Qext_sil[i,k]* pi* a[k]**2*B_sil[k,i]
            dI_em_sil[i, k] = dn_da_sil[k]* Qabs_sil[i,k]* pi* a[k]**2*B_sil[k,i] ##a. Thiem's correction
            dI_pol_sil[i, k] = dn_da_sil[k]* Qpol_sil[i,k]* pi* a[k]**2*B_sil[k,i]*fa[k]/2
            dI_pol_abs_sil[i, k] = dn_da_sil[k]* Qpol_abs_sil[i,k]* pi* a[k]**2*fa[k]
        
            #dI_em_amCBE[i, k] = dn_da_gra[k]* Qext_amCBE[i,k]* pi* a[k]**2*B_gra[k,i]
            dI_em_amCBE[i, k] = dn_da_gra[k]* Qabs_amCBE[i,k]* pi* a[k]**2*B_gra[k,i] ## a. Thiem's correction
            dI_pol_amCBE[i, k] = dn_da_gra[k]* Qpol_amCBE[i,k]* pi* a[k]**2*B_gra[k,i]*fa[k]/2
            dI_pol_abs_amCBE[i, k] = dn_da_gra[k]* Qpol_abs_amCBE[i,k]* pi* a[k]**2*fa[k]

        Iem_sil[i] = integrate.simps(dI_em_sil[i], a)
        Iem_amCBE[i] = integrate.simps(dI_em_amCBE[i], a)

        Ipol_sil[i] = integrate.simps(dI_pol_sil[i], a)
        Ipol_amCBE[i] = integrate.simps(dI_pol_amCBE[i], a)

        I_pol_abs_sil[i] = integrate.simps(dI_pol_abs_sil[i], a)
        I_pol_abs_amCBE[i] = integrate.simps(dI_pol_abs_amCBE[i], a)


    Iext = (Iem_amCBE +Iem_sil)* w
    Ipol_emis_sil  = Ipol_sil *w
    ratio_emis_sil = Ipol_emis_sil/Iext *100

    #dP_abs_sil = I_pol_abs_sil
    #P_abs_sil = dP_abs_sil*N_gas*100

    #The cas when only carbonaceous have aligned
    Ipol_emis_car  = Ipol_amCBE*w
    ratio_emis_car = Ipol_emis_car/Iext*100

    #The case when both sil and carbonaceous have been aligned
    Ipol_emis_tot  = (Ipol_sil+Ipol_amCBE)*w
    ratio_emis_tot = Ipol_emis_tot/Iext *100

    #dP_abs = I_pol_abs_sil +I_pol_abs_amCBE
    #P_abs = dP_abs*N_gas*100

    # ======================================================================
    # <<<<<====================>>>>> [OUTPUT] <<<<<====================>>>>>
    # ======================================================================
    # --------------- Pem ---------------
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

def plot_check():
    x1 = 7
    x2 = 6.e3
    y1 = -0.5
    y2 = 20.9
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
        if ratd == 'off':
            axis1.text(xax[1],yax[nyax-4],r'dust-type:%s'%(dust_type),weight="bold")
            axis1.text(xax[1],yax[nyax-7],r'no disruption',weight="bold")
            axis1.text(xax[1],yax[nyax-10],r'r = %3.1f'%(alpha))
        else:
            axis1.text(xax[1],yax[nyax-4],r'dust-type:%s'%(dust_type),weight="bold")
            axis1.text(xax[1],yax[nyax-7],r'RATD',weight="bold")
            axis1.text(xax[1],yax[nyax-10],r'Smax = 10$^%1d$ erg cm$^{-3}$'%(log10(Smax)))
            axis1.text(xax[1],yax[nyax-13],r'r = %3.1f'%(alpha))

        axis1.axis([x1,x2,y1,y2])
        axis1.set_xlabel('wavelength$\;(\mu$$m$)')
        axis1.set_ylabel('P$\;(\%)$')
        axis1.set_xscale('log')
        custm_axis(axis1,0.5,-0.06,-0.10,0.5)           
        fig1.savefig(dir_out+'Pem.pdf')
                
        fsil = interp1d(w,ratio_emis)
        percentage.append(fsil(working_lam))
    axis2.plot(Urange,percentage,'o-',color='black')
    #axis2.axis([x1,x2,y1,y2])
    
    if ratd == 'off':
        axis2.text(0.03,0.3,r'dust-type:%s'%(dust_type),weight="bold",va='center',transform=axis2.transAxes)
        axis2.text(0.03,0.22,r'no disruption',weight="bold",va='center',transform=axis2.transAxes)
        axis2.text(0.04,0.15,r'r = %3.1f'%(alpha),va='center',transform=axis2.transAxes)
    else:
        axis2.text(0.03,0.3,r'dust-type:%s'%(dust_type),weight="bold",va='center',transform=axis2.transAxes)
        axis2.text(0.03,0.22,r'RATD',weight="bold",va='center',transform=axis2.transAxes)
        axis2.text(0.03,0.15,r'Smax = 10$^%1d$ erg cm$^{-3}$'%(log10(Smax)),va='center',transform=axis2.transAxes)
        axis2.text(0.03,0.08,r'r = %3.1f'%(alpha),va='center',transform=axis2.transAxes)
        axis2.text(0.7,0.9,r'$\lambda$ = %3.1f$\, \mu$m'%(working_lam),va='center',transform=axis2.transAxes)
    
    axis2.set_xscale('log')
    axis2.set_xlabel('U')
    axis2.set_ylabel('P$\;(\%)$')
    fig2.savefig(dir_out+'Pem_U.pdf')
    #plt.show()

    os.system('open ' + dir_out+'Pem.pdf')
    os.system('open ' + dir_out+'Pem_U.pdf')
if __name__=='__main__':
    if (parallel):
        log.error('Sorry -- parallel calculation is being developed! [\033[1;5;7;91m failed \033[0m ]')
        raise IOError('parallel option')
    else: 
        for iu in range(0,len(Urange)):
            DustPol(iu)

    plot_check()