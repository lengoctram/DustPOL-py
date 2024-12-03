##import Built-in functions
import numpy as np
import os,re,time
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import warnings
import concurrent.futures
from astropy import log
from matplotlib.colors import LogNorm
from joblib import Parallel, delayed#, Memory
from scipy.interpolate import interp1d
from astropy.io import fits
from scipy import interpolate
from scipy.signal import savgol_filter

##import customized functions
from . import decorators 
from  .decorators import auto_refresh, printProgressBar, bcolors
from . import rad_func
from . import disrupt
from . import qq
from . import radiation
from . import align 
from . import size_distribution
from . import pol_degree
from . import DustPOL_io
from . import constants
from . import isoCloud_class
from . import analysis

##Ignore the warnings
warnings.filterwarnings('ignore')

class DustPOL:
    """This is the main routine of the DustPOL-py
        -- update_radiation     : check and update for radiation
        -- update_grain_size    : check and update for grain size
        -- size_distribution    : calling size distribution
        -- get_Planck_function  : calling Planck function
        Inputs:
        -------
            These args are passed from the input file
                + U,mean_lam,gamma, ##radiation
                + Tgas,Tdust,ngas,  ##physical conditions
                + ratd,Smax,        ##rotational disruption
                + amin,amax,power_index,dust_type,dust_to_gas_ratio,GSD_law, ##grain-size
                + RATalign,f_min,f_max,alpha,   ##alignment physics
                + B_angle                       ##B-field inclination
        Outputs:
        --------
            Some sub-routines to call-out
                + cal_pol_abs   : to compute the degree of absorption polarisation
                + cal_pol_emi   : to compute the degree of emisison polarisation
                + starless_r0   : to compute the degree of absorption and emission polarisations
                                :            of starless core at a given line of sight (r0 -- x-coorinate)
                + starless      : to compute the degree of absorption and emission polarisations in starless core 

        Examples:
        ---------
        - Starless core with the line of sight Av fixed for different amax
                starless(Av_fixed_amax=True,fixed_amax_value=1.e-4)
        - Starless core with the line of sight Av varried for different amax
                starless(Av_fixed_amax=False)
    """

    @auto_refresh
    def __init__(self,input_params_file):
        self.input_params_file = input_params_file
        params = DustPOL_io.input(self.input_params_file)
        # from DustPOL_io import (
        #     U, u_ISRF, mean_lam, gamma, Tgas, Tdust, ngas,
        #     ratd, Smax, amin, amax, power_index, rho, dust_type, 
        #     dust_to_gas_ratio, GSD_law, RATalign, f_max,alpha,
        #     B_angle, Bfield, Ncl, phi_sp, fp, rflat, rout,nsample,
        #     path, pc
        # )
        self.output_dir=params.output_dir
        self.U = params.U             #No-unit
        self.u_ISRF=params.u_ISRF     #erg cm-3
        self.gamma=params.gamma       #No-unit
        self.mean_lam=params.mean_lam #cm
        self.Tgas=params.Tgas         #K
        self.Tdust=params.Tdust       #K
        self.ngas=params.ngas         #cm-3
        self.ratd=params.ratd         #[option]
        self.Smax=params.Smax         #erg cm-3
        self.amin=params.amin         #cm
        self.amax=params.amax         #cm
        self.power_index      =params.power_index       #No-unit
        self.dust_type        =params.dust_type         #[option]
        self.dust_to_gas_ratio=params.dust_to_gas_ratio #No-unit
        self.GSD_law          =params.GSD_law           #[option]
        self.RATalign         =params.RATalign          #[option]
        self.Bfield = params.Bfield                     #[MRAT] -- otherwise, nan
        self.Ncl    = params.Ncl                        #[MRAT] -- otherwise, nan
        self.phi_sp = params.phi_sp                     #[MRAT] -- otherwise, nan
        self.fp     = params.fp                         #[MRAT] -- otherwise, nan
        self.f_min  = 0.0             #%
        self.f_max  = params.f_max    #%
        self.alpha  = params.alpha    #No-unit
        self.B_angle= params.B_angle  #radiant
        self.rho    = params.rho      #g cm-3
        self.parallel   =params.parallel       #[option] parallelization calculation
        if (self.parallel):
            self.max_workers=params.max_workers    #[if parallel]: numbers of CPU cores
        self.verbose=False
        self.Urange_tempdist=[]

        ##parameters for isolated cloud
        self.p     = params.p
        self.rflat = params.rflat #cm #17000.*constants.au.cgs.value
        self.rout  = params.rout  #cm #624.e6*constants.au.cgs.value
        self.nsample= params.nsample #int 5#50

        # # ------- get constants -------
        self.pc = constants.pc

        # # ------- get path to directory -------
        self.path=params.path

        # ------- Initialization wavelength, grain size, and cross-sections from the file -------
        if self.dust_type=='astro' or self.dust_type=='Astro':
            hdr_lines=4
            skip_lines=4
            len_a=169
            len_w=1129
            num_cols=8
            self.Data_aAstro=rad_func.readDC(self.path+'data/astrodust/Q_aAstro_%.3f'%(self.alpha)+'_P0.2_Fe0.00.DAT',hdr_lines,skip_lines,len_a,len_w,num_cols)

        else:
            #BELOW DOESN'T WORK FOR OBLATE SHAPE WITH S=2
            if float(self.alpha)==0.3333:
                hdr_lines = 4
                skip_lines=4
                len_a_sil=70
                len_a_car=100
                len_w=800
                num_cols=8
            elif float(self.alpha)==2.0:
                hdr_lines=4
                skip_lines=4
                len_a_sil=160
                len_a_car=160
                len_w=104
                num_cols=8
            else:
                log.error('Values of alpha is not regconized! [\033[1;5;7;91m failed \033[0m]')
            self.Data_sil = rad_func.readDC(self.path+'data/Q_aSil2001_'+str(self.alpha)+'_p20B.DAT',hdr_lines,skip_lines,len_a_sil,len_w,num_cols)
            self.Data_mCBE = rad_func.readDC(self.path+'data/Q_amCBE_'+str(self.alpha)+'.DAT',hdr_lines,skip_lines,len_a_car,len_w,num_cols)
        self.get_coefficients_files(verbose=self.verbose)
        
        # ------- Initialization grain-size distribution -------
        self.grain_size_distribution()  

    @auto_refresh
    def update_radiation(self):
        self.U = radiation.radiation_retrieve(self).retrieve()
        # self.U = rad.retrieve()

    @auto_refresh
    def update_grain_size(self,a,verbose=True):
        #update radiation
        #self.update_radiation()
        self.verbose=verbose

        if self.amax>max(a):
            raise ValueError('SORRY - amax should be %.5f [um] at most [\033[1;5;7;91m failed \033[0m]'%(max(a)*1e4))

        self.lmin = np.searchsorted(a, self.amin)
        self.lmax = np.searchsorted(a, min(self.amax, disrupt.radiative_disruption(self).a_disrupt(a))) if self.ratd else np.searchsorted(a, self.amax + 0.1 * self.amax)

        self.a = a[self.lmin:self.lmax]
        self.na = len(self.a*1e4)
        return

    @auto_refresh
    def dP_dT(self):
        ##This function reads the dust temperature distribution, pre-calculated
        qT = rad_func.T_dust(self.path, self.na, self.U)#T_dust(na,UINDEX)
        T_gra = qT[0]
        T_sil = qT[1]
        dP_dlnT_gra = qT[2]
        dP_dlnT_sil = qT[3]
        return T_sil,T_gra,dP_dlnT_sil,dP_dlnT_gra

    @auto_refresh
    def get_coefficients_files(self,verbose=True):
        ##This function reads the cross-sections, pre-calculated
        ##The outputs are 2d-array: a function of wavelength and grain-size
        if self.dust_type.lower()=='astro':
            self.w = self.Data_aAstro[1,:,0]*1e-4 ## wavelength in cm
            a = self.Data_aAstro[0,0,:]*1e-4      ## grain size in cm
            self.update_grain_size(a,verbose=verbose)             ## update grain size --> self.a
            [self.Qext_astro, self.Qabs_astro, self.Qpol_astro, self.Qpol_abs_astro] = qq.Qext_grain_astrodust(self.Data_aAstro,self.w,self.a,self.alpha)
            return
        else:
            self.w = rad_func.wave(self.path)     ##good for prolate shape
            a = rad_func.a_dust(self.path,10.0)[1] ##good for prolate shape
            self.update_grain_size(a,verbose=verbose)    ##update grain size
            
            if float(self.alpha)==2.0: ##efficiences data from POLARIS has a problem <-- fixed
                [self.Qext_sil, self.Qabs_sil, self.Qpol_sil, self.Qpol_abs_sil] = qq.Qext_grain(self.Data_sil,self.w,self.a,self.alpha,fixed=False,wmin=172e-4,wmax=628e-4,dtype='sil')        
                [self.Qext_amCBE, self.Qabs_amCBE, self.Qpol_amCBE, self.Qpol_abs_amCBE] = qq.Qext_grain(self.Data_mCBE,self.w,self.a,self.alpha,fixed=True,wmin=172e-4,wmax=300e-4,dtype='car')
            else:
                [self.Qext_sil, self.Qabs_sil, self.Qpol_sil, self.Qpol_abs_sil] = qq.Qext_grain(self.Data_sil,self.w,self.a,self.alpha)        
                [self.Qext_amCBE, self.Qabs_amCBE, self.Qpol_amCBE, self.Qpol_abs_amCBE] = qq.Qext_grain(self.Data_mCBE,self.w,self.a,self.alpha)
            return

    @auto_refresh
    def grain_size_distribution(self,fix_amax=False,fix_amax_value=None):
        ##This function compute the grain-size distribution
        ##The output is a 1d-array: a function of grain-size
        if self.dust_type.lower()=='astro':
            if not fix_amax:
                GSD_params = [self.a.min(),self.a.max(),2.74,self.dust_to_gas_ratio,self.power_index]
            else:
                GSD_params = [self.a.min(),fix_amax_value,2.74,self.dust_to_gas_ratio,self.power_index]
            self.dn_da_astro = size_distribution.dnda_astro(self.a,sizedist=self.GSD_law,MRN_params=GSD_params)
            return
        else:
            self.dn_da_gra = size_distribution.dnda(6,'carbon',self.a,self.GSD_law,self.power_index,self.dust_to_gas_ratio)
            self.dn_da_sil = size_distribution.dnda(6,'silicate',self.a,self.GSD_law,self.power_index,self.dust_to_gas_ratio)
            return

    @auto_refresh
    def get_Planck_function(self,Tdust,dP_dlnT=None):
        ##This function calculates the Planck-function
        ##The output is a 2d-array: a function of wavelength and grain-size
        if dP_dlnT is None:
            B_  = rad_func.planck_equi(self.w,self.na,Tdust) ##Tdust must be an array with 'na' element
        else:
            B_ = rad_func.planck(self.w,self.na,Tdust,dP_dlnT) ##function of U and na
        return B_

    @auto_refresh
    def extinction(self):
        ##This function return the extinction curve, normalized by Ngas
        if self.dust_type.lower()=='astro':
            dtau = self.Qext_astro * np.pi*self.a**2 * self.dn_da_astro
        elif self.dust_type.lower()=='sil':# in ['sil','silicate']:
            dtau = self.Qext_sil * np.pi*self.a**2 * self.dn_da_sil
        else:
            dtau_sil = self.Qext_sil * np.pi*self.a**2 * self.dn_da_sil
            dtau_car = self.Qext_amCBE * np.pi*self.a**2 * self.dn_da_gra
            dtau = dtau_sil + dtau_car
        
        tau_per_Ngas = integrate.simps(dtau,self.a)
        return 1.086*tau_per_Ngas

    @auto_refresh
    def cal_pol_abs(self,verbose=True,save_output=False):
        '''
        ###-------------------------------------------------------------------------------##
        ## This function calculates the degree of starlight polarization (0D)
        ## The inputs are taken from the input datafile
        ## The outputs are 1d-arrays: 
        ##      1- wavelength in cm 
        ##      2- pext/Ngas: for single dust composition (astrodust, silicate)        
        ##      3- pext/Ngas: for mixtured dust composition (silicate+carbon bonded together)
        ##           ***Note: if astrodust is used, pext/Ngas for mixture composition is nan 
        ## save_output: to write the ouptput out in a file at output folder
        ##-------------------------------------------------------------------------------##
        '''
        self.verbose=verbose #transfer verbose to parent var to pass to pol_degree and align classes
        
        if (self.verbose):
            print('\t \033[1;7;34m U=%.3f \033[0m   \t\t '%(self.U))
        w,dP_abs,dP_abs_mix = pol_degree.pol_degree(self)._pol_degree_absorption_(self)

        ##checking for negative values that might be resulted from the numerical cals
        # dP_abs[dP_abs<0]=np.nan
        # dP_abs_mix[dP_abs_mix<0]=np.nan
        
        # dP_abs = savgol_filter(dP_abs,20,2) # smooth (for visualization) -- not physically affected
        # dP_abs_mix = savgol_filter(dP_abs_mix,20,2) # smooth (for visualization) -- not physically affected

        if (save_output):
            #Save the output
            data_save={}
            data_save['wavelength (micron)'] = w*1e4
            if self.dust_type.lower()=='astro':
                data_save['p/Ngas (%/cm-2)'] = dP_abs/self.ngas
            else:
                data_save['p_sil/Ngas (%/cm-2)'] = dP_abs/self.ngas
                data_save['p_tot/Ngas (%/cm-2)'] = dP_abs_mix/self.ngas
            ext_curve = self.extinction()
            data_save['A/Ngas']=ext_curve
            fileout = 'starlight.dat'
            DustPOL_io.output(self,fileout,data_save)
        return w,dP_abs,dP_abs_mix#/self.ngas

    @auto_refresh
    def cal_pol_emi(self,tau=0.0,Tdust=None,verbose=True,save_output=False):#,get_planck_option=True):
        '''
        ##-------------------------------------------------------------------------------##
        ## This function calculates the degree of thermal dust polarization (0D)
        ## The inputs are taken from the input datafile
        ## The output is a 1d-array: a function of wavelength
        ##      If dust_type is Astrodust (astro):
        ##          return [total intensity, polarized intensity, zeros_array], [pol. degree, zeros_array]
        ##      If dust_type is silicate (sil):
        ##          return [total intensity, polarized intensity of sil, polarized intensity of sil+car], 
        ##              [pol. degree of sil, pol. degree of sil+car]
        ## save_output: to write the ouptput out in a file at output folder
        ##-------------------------------------------------------------------------------##
        '''
        self.verbose=verbose #transfer verbose to parent var to pass to pol_degree and align classes

        if self.dust_type.lower()=='astro':
            if Tdust is None:
                Tdust = 16.4* self.U**(1./6) * (self.a/1.e-5)**(-1./15)#* np.ones(self.na)
                if (self.verbose):
                    log.info('\033[1;7;34m U=%.3f : radiation -->> Tdust \033[0m   \t\t '%(self.U))
            elif isinstance(Tdust,(float,int)):
                if (self.verbose):
                    log.info('\033[1;7;34m U=%.3f and Tdust=%.3f (K) \033[0m   \t\t '%(self.U,Tdust))
                Tdust = float(Tdust) * (self.a/1.e-5)**(-1./15)
            self.B_astro=self.get_Planck_function(Tdust)
        else:
            T_sil,T_gra,dP_dlnT_sil,dP_dlnT_gra = self.dP_dT() ##function of U and na
            self.B_gra=self.get_Planck_function(T_gra,dP_dlnT_gra)
            self.B_sil=self.get_Planck_function(T_sil,dP_dlnT_sil)

        w,I_list,P_list = pol_degree.pol_degree(self)._pol_degree_emission_(self)
        
        if (save_output):
            #Save the output
            data_save={}
            data_save['wavelength (micron)'] = w*1e4
            if self.dust_type.lower()=='astro':
                data_save['Iem'] = I_list[0]
                data_save['Ipol'] = I_list[1]
                data_save['p (%)'] = P_list[0]

            else:
                data_save['Iem'] = I_list[0]
                data_save['Ipol_sil'] = I_list[1]
                data_save['Ipol_tot'] = I_list[2]
                data_save['p_sil (%)'] = P_list[0]
                data_save['p_tot(%)'] = P_list[1]

            fileout = 'thermal.dat'
            DustPOL_io.output(self,fileout,data_save)

        return w,I_list,P_list

    
    @auto_refresh
    def isoCloud_los(self,r0,Av_fixed_amax=False,fixed_amax_value=None,progress=False,get_info=True,save_output=False,filename_output=None):
        '''
        ##-------------------------------------------------------------------------------##
        ## This function calculates the degree of starlight and thermal dust polarization (1D)
        ## For the fundamentals, see [website] for insights
        ## The integration along a given line of sight 'r0'
        ## Av_fixed_amax[option]: to compare the pol. spectrum for different amax
        ##              -- Av is sensitive to amax --> differetn amax results in distinc Av
        ##              -- to get rid of this Av-dependency, set av_fixed_amax for all amax
        ## fixed_amax_value[option]: if Av_fixed_amax=True, please give the value of amax to anchor
        ## If the option save_out==True, the output will be written out in the ascii files
        ##                              --> the name of these files can be customized with the option filename_output
        ##                              --> the exanpsion will be added as: filename_output_abs.dat and filename_output_emi.dat
        ##                              [!] if filename_output is not given: the default names p_abs.dat and p_emi.dat
        ##-------------------------------------------------------------------------------##
        '''
        if (get_info):
            if (progress):
                self.verbose=False
            else:
                self.verbose=True
        else:
            self.verbose=False

        ##Re-Initialization the radiation, grain size, coefficients and distribution --> to update "self" global parameters
        self.update_radiation()
        self.get_coefficients_files(verbose=self.verbose) ##need to be here <-- grain size 
        self.grain_size_distribution(fix_amax=Av_fixed_amax,fix_amax_value=fixed_amax_value)##must be after get_coefficient_files

        U_0=self.U          ##hard copy of the initial radiation field
        ngas_0=self.ngas    ##hard copy of the initial gas volume density

        #call the starless_profile
        isoCloud_exe = isoCloud_class.isoCloud_profile()#(self)
        coords,rr=isoCloud_exe.isoCloud_model(self) #the global parameter "self" has been updated...
        x_,y_,z_=coords
        max_radius = rr.max()

        Av_ = isoCloud_exe.Av_func(self,r0)

        if get_info:
            print('-----------Get ngas------------------')
            log.info('ngas_0=%.3e (cm-3)'%self.ngas)                

            print('-----------Get distance params-------')
            log.info('rflat=%.3f (pc)'%(self.rflat/self.pc))
            log.info('max_radius=%.3f (pc)'%(max_radius/self.pc))

            print('-----------Get observed Av-----------')
            log.info('Av_los=%.3f (mag.)'%Av_)

        if (len(z_)<2*self.nsample+1):
            raise IOError('descritation of z axis is wrong!')

        # get_planck_option=True

        dp_abs_matrix = np.zeros((len(z_),len(self.w)))
        dpmix_abs_matrix = np.zeros((len(z_),len(self.w)))

        dIext_emi_matrix = np.zeros((len(z_),len(self.w)))
        dIp_emi_matrix = np.zeros((len(z_),len(self.w)))
        dIpmix_emi_matrix = np.zeros((len(z_),len(self.w)))

        nH_=np.zeros(len(z_))
        Av_compute=np.zeros(len(z_))
        ali_=np.zeros(len(z_))

        ##loop over z_ along a single LOS
        if (progress) & (get_info): 
            printProgressBar(0, len(z_), prefix = '  -> Progress:', suffix = 'Complete', length = 30)

        for j in range(len(z_)):
            ##re-update the grain size, coefficients and distribution
            ## (e.g., disruption might happen along the LOS "z") --> to update "self"
            self.get_coefficients_files(verbose=self.verbose)
            self.grain_size_distribution(fix_amax=Av_fixed_amax,fix_amax_value=fixed_amax_value)

            r_compute=np.sqrt(z_[j]*z_[j]+r0*r0)
            if (r_compute>max_radius):
                # self.ngas=np.nan
                dp_abs_matrix[j,:] =np.zeros(len(self.w))
                dpmix_abs_matrix[j,:] =np.zeros(len(self.w))
                dIext_emi_matrix[j,:] =np.zeros(len(self.w))
                dIp_emi_matrix[j,:]=np.zeros(len(self.w))
                dIpmix_emi_matrix[j,:]=np.zeros(len(self.w))

                nH_[j]=0.0
                Av_compute[j]=0.0
                ali_[j]=0.0
                if (get_info):
                    if (progress):
                        printProgressBar(j+1, len(z_), prefix = '  -> Progress:', suffix = 'Complete', length = 30)
                    else:
                        print('z_=%.3f (pc), Av_los=%.3f (mag), ngas=%.3e'%(z_[j]/self.pc,Av_,nH_[j]))
                continue        
            else:
                #print('-----------Get the local computed Av-----------')
                Av_compute[j]=isoCloud_exe.Av_2calcule(ngas_0,self.rflat,self.p,Rv=4.0)(r_compute)

                #get U from starless law
                self.U = isoCloud_exe.U_starless(U_0,Av_compute[j])
                self.update_radiation()

                #print('-----------Get Tgas-----------')
                #print('Tgas_init=',self.Tgas)
                self.Tgas=isoCloud_exe.Tgas_starless(U_0,Av_compute[j],16.4)

                #print('-----------Get Dust-----------')
                self.Tdust=isoCloud_exe.Tdust_starless(U_0,Av_compute[j],16.4,self.a)

                #print('-----------Get mean_lam-----------')
                #print('mean_lam_init=',self.mean_lam*1e4)
                self.mean_lam=isoCloud_exe.lamda_starless(1.3e-4,Av_compute[j])

                # self.ngas=starless_exe.ngas_starless(ngas_0,self.rflat)(np.sqrt(z_[j]*z_[j]+r0*r0))                
                self.ngas=isoCloud_exe.ngas_starless(ngas_0,self.rflat,self.p)(r_compute) 
                # dtau=starless_exe.get_dtau(self,self.ngas)
                # dtau_850 = interp1d(self.w,dtau,axis=0)(850e-4)*abs(z_[j])  

                if (get_info):
                    if (progress):
                        printProgressBar(j+1, len(z_), prefix = '  -> Progress:', suffix = 'Complete', length = 30)
                    else:
                        print('\n')
                        log.info('z_=%.3e (pc), Av_los=%.3f (mag), Av_compute=%.3f, U=%.3f, ngas=%.3e, amax=%.3f \t\t'%(z_[j]/self.pc,Av_,Av_compute[j],self.U,self.ngas,self.a.max()*1e4))

                #absorption polarization
                w_abs,dp_abs,dpmix_abs=self.cal_pol_abs(verbose=self.verbose)
                
                #emission polarization
                dIext_emi,dIp_emi,dIpmix_emi=self.cal_pol_emi(Tdust=self.Tdust,verbose=self.verbose)[1]

                Iext_850 = interp1d(self.w,dIext_emi,axis=0)(850e-4)
                Ipsil_850= interp1d(self.w,dIp_emi,axis=0)(850e-4)
                dp_abs_matrix[j,:]   =dp_abs
                dpmix_abs_matrix[j,:]=dpmix_abs
               
                dIext_emi_matrix[j,:] =dIext_emi
                dIp_emi_matrix[j,:]   =dIp_emi
                dIpmix_emi_matrix[j,:]=dIpmix_emi

                nH_[j]=self.ngas


        NH_=integrate.simps(nH_,z_)
        p_abs   = integrate.simps(dp_abs_matrix,z_,axis=0) 
        pmix_abs= integrate.simps(dpmix_abs_matrix,z_,axis=0)
        # p_abs[np.where(p_abs<1e-3)]=0.0
        # pmix_abs[np.where(pmix_abs<1e-3)]=0.0

        Iext_emi = integrate.simps(dIext_emi_matrix,z_,axis=0)
        Ip_emi   = integrate.simps(dIp_emi_matrix,z_,axis=0)
        Ipmix_emi= integrate.simps(dIpmix_emi_matrix,z_,axis=0)
        p_emi    = Ip_emi/Iext_emi*100
        pmix_emi = Ipmix_emi/Iext_emi*100
        # psil_850 = interp1d(self.w,psil_emi,axis=0)(850e-4)
        # print('             **** psil_emi(850)=%.3f'%(psil_850))
        if (save_output):
            if (filename_output is None):
                output_abs = 'p_abs.dat'
                output_emi = 'p_emi.dat'
            else:
                output_abs = filename_output+'_abs.dat'
                output_emi = filename_output+'_emi.dat'
            data_abs={}
            data_emi={}    

            data_abs['w']=self.w*1e4
            data_emi['w']=self.w*1e4
        
            data_abs['pabs']=p_abs
            data_abs['pabs_mix']=pmix_abs

            data_emi['pemi']=p_emi
            data_emi['pemi_mix']=pmix_emi

            data_emi['Iext']=Iext_emi

            self.Av_array=Av_
            self.__init__(self.input_params_file) ##reset the initial parameters
            DustPOL_io.output(self,output_abs,data_abs)
            DustPOL_io.output(self,output_emi,data_emi)

        return self.w,NH_,Av_,Iext_emi,[p_abs,pmix_abs],[p_emi,pmix_emi]

    @auto_refresh
    def isoCloud_pos(self,Av_fixed_amax=False,fixed_amax_value=None,progress=False):
        '''
        ##-------------------------------------------------------------------------------##
        ## This function calculates the degree of starlight and thermal dust polarization (2D)
        ## For the fundamentals, see [website] for insights
        ## The integration along a given line of sight 'r0'
        ## Av_fixed_amax[option]: to compare the pol. spectrum for different amax
        ##              -- Av is sensitive to amax --> differetn amax results in distinc Av
        ##              -- to get rid of this Av-dependency, set av_fixed_amax for all amax
        ## fixed_amax_value[option]: if Av_fixed_amax=True, please give the value of amax to anchor
        ##-------------------------------------------------------------------------------##
        '''
        self.get_coefficients_files(verbose=False) ##redundant (because it's called in __init__), but just to be sure ...
        isoCloud_exe = isoCloud_class.isoCloud_profile()
        coords,max_radius=isoCloud_exe.isoCloud_model(self)
        x_,y_,z_=coords

        data_abs={};output_abs = 'p_amax=%.2f'%(self.amax*1e4)+'_abs.dat'
        data_abs['w']=self.w*1e4
        
        data_mix_abs={};output_mix_abs = 'pmix_amax=%.2f'%(self.amax*1e4)+'_abs.dat'
        data_mix_abs['w']=self.w*1e4

        data_emi={};output_emi = 'p_amax=%.2f'%(self.amax*1e4)+'_emi.dat'
        data_emi['w']=self.w*1e4
        
        data_mix_emi={};output_mix_emi = 'pmix_amax=%.2f'%(self.amax*1e4)+'_emi.dat'
        data_mix_emi['w']=self.w*1e4

        Av_array=[];x_=x_[x_>=0]
        r0_range=x_[::2]#np.linspace(0,self.rout/2e3,30)#*constants.pc.cgs.value

        # Av_test=np.zeros((len(r0_range),len(z_)))
        # ali_test=np.zeros((len(r0_range),len(z_)))
        start_time=time.time()

        if (not self.parallel): #None parallelization
            for i,r0 in enumerate(r0_range):
                print('---------------------------------------------------')
                print('cell number=%d/%d'%(i,len(r0_range)), 'r0=%.3e (pc)'%(r0/self.pc))

                self.__init__(self.input_params_file) ##reset the initial parameters
            
                w,NH_,Av_,Iext_,pabs,pemi=self.isoCloud_los(
                                        r0,
                                        Av_fixed_amax=Av_fixed_amax,
                                        fixed_amax_value=fixed_amax_value,
                                        progress=progress
                                        )
                Av_array.append(Av_)

                data_abs['p(Av=%.3f)'%Av_]=pabs[0]
                data_mix_abs['p(Av=%.3f)'%Av_]=pabs[1]

                data_emi['p(Av=%.3f)'%Av_]=pemi[0]
                data_mix_emi['p(Av=%.3f)'%Av_]=pemi[1]

                data_emi['I(Av=%.3f)'%Av_]=Iext_
                data_mix_emi['I(Av=%.3f)'%Av_]=Iext_

        else: #parallelization
            get_info=False
            progress=False
            log.info('Parallel computation with : \033[1;36m %d \033[0m CPU cores'%(self.max_workers))
            # printProgressBar(0, len(r0_range), prefix = '  -> Submit and Process  :', suffix = 'Complete', length = 30,line=2)
            # printProgressBar(0, len(r0_range), prefix = '  -> Process the Complete:', suffix = 'Complete', length = 30,line=1)
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit tasks to the executor
                j_submit=0
                j_process=0
                futures = []
                for r0 in r0_range:
                    # Reset initial parameters
                    self.__init__(self.input_params_file)
                    try:
                        future = executor.submit(
                            self.isoCloud_los,
                            r0,
                            Av_fixed_amax=Av_fixed_amax,
                            fixed_amax_value=fixed_amax_value,
                            progress=progress,
                            get_info=get_info
                        )
                        if future is not None:
                            futures.append(future)
                            j_submit=j_submit+1
                            printProgressBar(j_submit, len(r0_range), prefix = '  -> Submit and Process  :', suffix = 'Complete', length = 30)
                        else:
                            print(f"Warning: executor.submit returned None for r0={r0}")
                    except Exception as e:
                        print(f"Error submitting task for r0={r0}: {e}")
                # Ensure no NoneType in futures
                if not futures:
                    raise RuntimeError("No valid futures were created. Check your task submission logic.")

                # Process completed futures                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()  # Retrieve the result of the future
                        # print('Futures were created!')
                        w,NH_,Av_,Iext_,pabs,pemi=result

                        Av_array.append(Av_)

                        data_abs['p(Av=%.3f)'%Av_]=pabs[0]
                        data_mix_abs['p(Av=%.3f)'%Av_]=pabs[1]

                        data_emi['p(Av=%.3f)'%Av_]=pemi[0]
                        data_mix_emi['p(Av=%.3f)'%Av_]=pemi[1]

                        data_emi['I(Av=%.3f)'%Av_]=Iext_
                        data_mix_emi['I(Av=%.3f)'%Av_]=Iext_

                        j_process=j_process+1            
                        printProgressBar(j_process, len(r0_range), prefix = '  -> Process the Complete:', suffix = 'Complete', length = 30)

                    except Exception as e:
                        print(f"Task generated an exception: {e}")


        self.Av_array=np.array(Av_array)
        self.__init__(self.input_params_file) ##reset the initial parameters
        #There is a draw back of this saveout method: 
        #  If the keys are the same (two exact value of Av)
        #  Save the last array!!!
        DustPOL_io.output(self,output_abs,data_abs)
        DustPOL_io.output(self,output_mix_abs,data_mix_abs)
            
        DustPOL_io.output(self,output_emi,data_emi)
        DustPOL_io.output(self,output_mix_emi,data_mix_emi)
        end_time=time.time()
        if end_time-start_time<60:
            log.info('  -> Time for execution is %.2f secs'%(end_time-start_time))
        elif end_time-start_time<3600:
            print('  -> Time for execution is %.2f mins'%((end_time-start_time)/60))
        else:
            print('  -> Time for execution is %.2f hrs'%((end_time-start_time)/60/60))
