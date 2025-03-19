import numpy as np
import os,re
from . import rad_func, align, disrupt, qq#, starless_class, size_distribution
from astropy import log, constants
# import pol_degree, radiation
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib.colors import LogNorm
from  .decorators import auto_refresh

class pol_degree(object):
    # def __init__(self,
    #                 U,mean_lam,gamma, ##radiation
    #                 Tgas,Tdust,ngas, ##physical conditions
    #                 ratd,Smax, 	##rotational disruption
    #                 amin,amax,power_index,dust_type,dust_to_gas_ratio,GSD_law, ##grain-size
    #                 RATalign,f_min,f_max,alpha, 		##alignment physics
    #                 B_angle			##B-field inclination
    #             ):
    def __init__(self,parent):
        self.U = parent.U
        self.gamma=parent.gamma
        self.mean_lam=parent.mean_lam
        self.Tgas=parent.Tgas
        self.Tdust=parent.Tdust
        self.ngas=parent.ngas
        self.ratd=parent.ratd
        self.Smax=parent.Smax
        self.amin=parent.amin
        self.amax=parent.amax
        self.a   =parent.a
        self.na  =parent.na
        self.power_index=parent.power_index
        self.dust_type=parent.dust_type
        self.dust_to_gas_ratio=parent.dust_to_gas_ratio
        self.GSD_law=parent.GSD_law
        self.RATalign=parent.RATalign
        self.Bfield=parent.Bfield
        self.Ncl=parent.Ncl
        self.phi_sp=parent.phi_sp
        self.fp=parent.fp
        self.f_min=parent.f_min
        self.f_max=parent.f_max
        # self.alpha=parent.alpha
        self.B_angle=parent.B_angle
        self.Urange_tempdist=[]
        self.u_ISRF=parent.u_ISRF
        self.rho=parent.rho
        self.w  = parent.w
        self.verbose=parent.verbose
        # self.progress=parent.progress
        # self.B_gra=parent.B_gra
        # self.B_sil=parent.B_sil
        # self.Data_sil=parent.Data_sil
        # self.Data_mCBE=parent.Data_mCBE

        # self.Qext_sil=parent.Qext_sil
        # self.Qabs_sil=parent.Qabs_sil
        # self.Qpol_sil=parent.Qpol_sil
        # self.Qpol_abs_sil=parent.Qpol_abs_sil

        # self.Qext_amCBE=parent.Qext_amCBE
        # self.Qabs_amCBE=parent.Qabs_amCBE
        # self.Qpol_amCBE=parent.Qpol_amCBE
        # self.Qpol_abs_amCBE=parent.Qpol_abs_amCBE


        # ##test starless core
        # # self.n0_gas=1e5
        # self.rflat = parent.rflat
        # self.rout=parent.rout
        # self.nsample=parent.nsample
        # log.info('self.U=%.3f'%(self.U))

        ##get the grain size distribution

        # # cal=pol_degree.pol_degree(self)

        # ##get the grain size distribution
        # self.dn_da_sil,self.dn_da_gra=self.grain_size_distribution()
        # print('self.na=',self.na,len(self.dn_da_sil),len(self.dn_da_gra))
        # log.info('max(a)=%.2f (um)'%(self.a.max()*1e4))
        # # cal=pol_degree.pol_degree(self)

        # # ##get the Tgrain distribution
        # # self.T_sil,self.T_gra,self.dP_dlnT_sil,self.dP_dlnT_gra = self.dP_dT()

        ##get the alignment size and alignment function
        ali_cl = align.alignment_class(self)
        a_ali  = ali_cl.Aligned_Size_v2()
        # if not (self.progress):
        #     log.info(' [Cross-check] align=%.3f \u2713 '%(a_ali*1e4))
        self.fa = ali_cl.f_ali()

        # ##-------- new dust temperature and its probability distribution --------
        # self.T_gra = T_gra[self.lmin:self.lmax+1, :]
        # self.T_sil = T_sil[self.lmin:self.lmax+1, :]
        # self.dP_dlnT_gra = dP_dlnT_gra[self.lmin:self.lmax+1, :]
        # self.dP_dlnT_sil = dP_dlnT_sil[self.lmin:self.lmax+1, :]

        # -------- PLANCK FUNCTION -------- 
        ##This one is preferable, but the computation costs quite a bit
        # self.B_gra = rad_func.planck(self.w,self.na,self.T_gra,self.dP_dlnT_gra)
        # self.B_sil = rad_func.planck(self.w,self.na,self.T_sil,self.dP_dlnT_sil)

        # -------- efficiency factors : Qext, Qpol --------
        # self.get_coefficients_files()
        # if float(self.alpha)==2.0: ##efficiences data from POLARIS has a problem <-- fixed
        #     [self.Qext_sil, self.Qabs_sil, self.Qpol_sil, self.Qpol_abs_sil] = qq.Qext_grain(self.Data_sil,self.w,self.a,self.alpha,fixed=False,wmin=172e-4,wmax=628e-4,dtype='sil')        
        #     [self.Qext_amCBE, self.Qabs_amCBE, self.Qpol_amCBE, self.Qpol_abs_amCBE] = qq.Qext_grain(self.Data_mCBE,self.w,self.a,self.alpha,fixed=True,wmin=172e-4,wmax=300e-4,dtype='car')
        # else:
        #     [self.Qext_sil, self.Qabs_sil, self.Qpol_sil, self.Qpol_abs_sil] = qq.Qext_grain(self.Data_sil,self.w,self.a,self.alpha)        
        #     [self.Qext_amCBE, self.Qabs_amCBE, self.Qpol_amCBE, self.Qpol_abs_amCBE] = qq.Qext_grain(self.Data_mCBE,self.w,self.a,self.alpha)

    @auto_refresh
    def _pol_degree_absorption_(self,parent):
        if self.dust_type.lower()=='astro' or self.dust_type.lower()=='astro+pah':
            #For the moment: PAH is not aligned and produced polarization
            dn_da_astro=parent.dn_da_astro
            Qpol_abs_astro=parent.Qpol_abs_astro

            dP_abs_a = dn_da_astro* Qpol_abs_astro* np.pi* self.a**2 * self.fa
            dP_abs   = integrate.simps(dP_abs_a, self.a) * self.ngas*100#* exp(-tau)
            return self.w,dP_abs,np.zeros(len(self.w))

        else:
            dn_da_sil=parent.dn_da_sil
            dn_da_gra=parent.dn_da_gra

            Qpol_abs_sil=parent.Qpol_abs_sil
            Qpol_abs_amCBE=parent.Qpol_abs_amCBE

            dP_abs_sil_a = dn_da_sil* Qpol_abs_sil* np.pi* self.a**2 * self.fa
            dP_abs_sil   = integrate.simps(dP_abs_sil_a, self.a) * self.ngas*100#* exp(-tau)

            dP_abs_car_a = dn_da_gra* Qpol_abs_amCBE* np.pi* self.a**2 * self.fa
            dP_abs_car   = integrate.simps(dP_abs_car_a, self.a)*self.ngas*100 #* exp(-tau)

            dP_abs_mix     = dP_abs_sil+dP_abs_car
            return self.w,dP_abs_sil,dP_abs_mix

    @auto_refresh
    def _pol_degree_emission_(self,parent,tau=0.0):
        if self.dust_type.lower()=='astro' or self.dust_type.lower()=='astro+pah':
            #For the moment: PAH is not aligned and produced polarization

            ## get globals argument to pass into functions fIem, fIpol
            self.dn_da_astro=parent.dn_da_astro 
            self.B_astro=parent.B_astro
            self.Qabs_astro=parent.Qabs_astro
            self.Qpol_astro=parent.Qpol_astro

            self.fIem_astro(tau)  ## return in global Iem_astro
            self.fIpol_astro(tau) ## return in global Ipol_astro

            Iext_astro = self.Iem_astro* self.w
            Ipol_emis_astro=self.Ipol_astro*self.w
            P_em_astro=Ipol_emis_astro/Iext_astro*100
            return self.w,[Iext_astro,Ipol_emis_astro,np.zeros(len(self.w))],[P_em_astro,np.zeros(len(self.w))]

        else:
            ## get globals argument to pass into functions fIem, fIpol
            self.dn_da_sil=parent.dn_da_sil
            self.dn_da_gra=parent.dn_da_gra

            self.B_gra=parent.B_gra
            self.B_sil=parent.B_sil

            self.Qext_sil=parent.Qext_sil
            self.Qabs_sil=parent.Qabs_sil
            self.Qpol_sil=parent.Qpol_sil

            self.Qext_amCBE=parent.Qext_amCBE
            self.Qabs_amCBE=parent.Qabs_amCBE
            self.Qpol_amCBE=parent.Qpol_amCBE

            # -------- Total emission --------
            #Total emission intensity, produced by both Sil and Carb
            self.fIem_sil(tau)
            self.fIem_car(tau)
            Iext = (self.Iem_amCBE +self.Iem_sil)* self.w
            
            # -------- Polarized emission --------
            #If only Sil are aligned, Ipol = Ipol_Sil
            self.fIpol_sil(tau)
            Ipol_emis_sil  = self.Ipol_sil *self.w
            P_em_sil = Ipol_emis_sil/Iext *100

            #The case when only carbonaceous are aligned
            self.fIpol_car(tau)
            Ipol_emis_car  = self.Ipol_amCBE*self.w
            P_em_car = Ipol_emis_car/Iext*100

            #The case when both sil and carbonaceous have been aligned
            Ipol_emis_tot  = (self.Ipol_sil + self.Ipol_amCBE)*self.w
            P_em_mix = Ipol_emis_tot/Iext *100
            return self.w,[Iext,Ipol_emis_sil,Ipol_emis_tot],[P_em_sil,P_em_mix]

    # # -------- optical depth --------
    # def optical_depth(self):
    #     #optical depth of silicate
    #     fsil_ext= self.Qext_sil * np.pi *self.a*self.a * self.dn_da_sil * NH
    #     tau_sil = integrate.simps(fsil_ext, a)
    #     #optical depth of carbon
    #     fgra_ext= Qext_amCBE * pi *a*a * dn_da_gra* NH
    #     tau_gra = integrate.simps(fgra_ext, a)
    #     tau = tau_sil+tau_gra

    # ==================== Polarized emission ====================
    def fIem_sil(self,tau):
        # -------- Silicate grain --------
        ##Total intensities
        arr1=self.ngas*self.dn_da_sil* (self.Qabs_sil+self.Qpol_sil*self.fa*(2./3-np.sin(self.B_angle)*np.sin(self.B_angle)))* np.pi* self.a**2 
        arr2=self.B_sil.T * np.array([np.exp(-tau)]).T
        dI_em_sil=np.multiply(arr1,arr2)
        self.Iem_sil = integrate.simps(dI_em_sil, self.a)
        return

    def fIpol_sil(self,tau):
        ##Polarized emission intensities
        arr2=self.B_sil.T * np.array([np.exp(-tau)]).T
        arr3=self.ngas*self.dn_da_sil* self.Qpol_sil* np.pi* self.a**2 * self.fa/2 *np.sin(self.B_angle)*np.sin(self.B_angle)
        dI_pol_sil=np.multiply(arr3,arr2)
        self.Ipol_sil = integrate.simps(dI_pol_sil, self.a)
        return

        # ##Polarized absorption intensities
        # dI_pol_abs_sil = dn_da_sil* Qpol_abs_sil* pi* a**2 * fa
        # Ipol_abs_sil   = integrate.simps(dI_pol_abs_sil, a) #* exp(-tau)

    def fIem_car(self,tau):
        # -------- Carbonaceous grain --------
        ##Total intensities
        arr1=self.ngas*self.dn_da_gra* self.Qabs_amCBE* np.pi* self.a**2
        arr2=self.B_gra.T * np.array([np.exp(-tau)]).T
        dI_em_amCBE=np.multiply(arr1,arr2)
        self.Iem_amCBE  = integrate.simps(dI_em_amCBE, self.a)
        return

    def fIpol_car(self,tau):
        ##Polarized emission intensities
        arr2=self.B_gra.T * np.array([np.exp(-tau)]).T
        arr3=self.ngas*self.dn_da_gra* self.Qpol_amCBE* np.pi* self.a**2 * self.fa/2
        dI_pol_amCBE=np.multiply(arr3,arr2)
        self.Ipol_amCBE = integrate.simps(dI_pol_amCBE, self.a)
        return
        # ##Polarized absorption intensities
        # dI_pol_abs_amCBE = dn_da_gra* Qpol_abs_amCBE* pi* a**2 * fa
        # Ipol_abs_amCBE   = integrate.simps(dI_pol_abs_amCBE, a) #* exp(-tau)

    def fIem_astro(self,tau):
        # -------- Silicate grain --------
        ##Total intensities
        # arr1=self.ngas*self.dn_da_astro* (self.Qabs_astro+self.Qpol_astro*self.fa*(2./3-np.sin(self.B_angle)*np.sin(self.B_angle)))* np.pi* self.a**2 
        arr1=self.ngas*self.dn_da_astro* self.Qabs_astro* np.pi* self.a**2
        arr2=self.B_astro.T * np.array([np.exp(-tau)]).T
        dI_em_astro=np.multiply(arr1,arr2)
        self.Iem_astro = integrate.simps(dI_em_astro, self.a)
        return

    def fIpol_astro(self,tau):
        ##Polarized emission intensities
        arr2=self.B_astro.T * np.array([np.exp(-tau)]).T
        arr3=self.ngas*self.dn_da_astro* self.Qpol_astro* np.pi* self.a**2 * self.fa *np.sin(self.B_angle)*np.sin(self.B_angle)
        dI_pol_astro=np.multiply(arr3,arr2)
        self.Ipol_astro = integrate.simps(dI_pol_astro, self.a)
        return

    # # ------- dust grain size -------
    # def grain_size_distribution(self):
    #     # #a = rad_func.a_dust(UINDEX)[1]
    #     # a = rad_func.a_dust(10.0)[1] ##good for prolate shape
    #     # # a = Data_sil[0,0,:]*1e-4; a=a[where(a<=amax)[0]]
    #     # if self.amax>max(a):
    #     #     log.error('SORRY - amax should be %.5f [um] at most [\033[1;5;7;91m failed \033[0m]'%(max(a)*1e4))
    #     #     raise IOError('Value of amax!')
    #     # na = len(a)
    #     # print('[input]na=',na)
    #     # #

    #     # # ------- update grain size -------
    #     # self.lmin = min(np.where(a>=self.amin)[0])
        
    #     # if (self.ratd == 'on'):
    #     #     disruption_ = disrupt.radiative_disruption(self) 
    #     #     a_disr = disruption_.a_disrupt(a)
    #     #     if a_disr>self.amax:
    #     #         self.lmax=abs(np.log10(a)-np.log10(self.amax)).argmin()
    #     #     else:
    #     #         self.lmax = abs(np.log10(a)-np.log10(a_disr)).argmin()
    #     #     # lmax = abs(log10(a)-log10(a_disr)).argmin()
    #     # if (self.ratd == 'off'):
    #     #     self.lmax = max(np.where(a<=self.amax+0.1*self.amax)[0])
    #     # a = a[self.lmin:self.lmax+1]
    #     # na = len(a)
    #     log.info('na=%d'%self.na)
    #     if self.dust_type=='astro' or self.dust_type=='Astro':
    #         dn_da_astro = size_distribution.dnda_astro(self.a)
    #         return dn_da_astro
    #     else:
    #         dn_da_gra = size_distribution.dnda(6,'carbon',self.a,self.GSD_law,self.power_index,self.dust_to_gas_ratio)
    #         dn_da_sil = size_distribution.dnda(6,'silicate',self.a,self.GSD_law,self.power_index,self.dust_to_gas_ratio)
    #         return dn_da_sil, dn_da_gra


    # def get_coefficients_files(self):
    #     #BELOW DOESN'T WORK FOR OBLATE SHAPE WITH S=2
    #     if float(self.alpha)==0.3333:
    #         hdr_lines = 4
    #         skip_lines=4
    #         len_a_sil=70
    #         len_a_car=100
    #         len_w=800
    #         num_cols=8
    #     elif float(self.alpha)==2.0:
    #         hdr_lines=4
    #         skip_lines=4
    #         len_a_sil=160
    #         len_a_car=160
    #         len_w=104
    #         num_cols=8
    #     else:
    #         log.error('Values of alpha is not regconized! [\033[1;5;7;91m failed \033[0m]')
    #     self.Data_sil = rad_func.readDC('./data/Q_aSil2001_'+str(self.alpha)+'_p20B.DAT',hdr_lines,skip_lines,len_a_sil,len_w,num_cols)
    #     self.Data_mCBE = rad_func.readDC('./data/Q_amCBE_'+str(self.alpha)+'.DAT',hdr_lines,skip_lines,len_a_car,len_w,num_cols)
    #     return

# class 
#     def starless_core(self):
#         fig,ax=plt.subplots()
#         phys_ = starless_class.starless_profile(self)
#         # Av_   = phys_.get_Av()
#         # align_=phys_.get_align()
#         # im = plt.imshow(align_.T*1e4,interpolation='bilinear',origin='lower',cmap='magma',norm=LogNorm())
#         # cbar=plt.colorbar(im,ax=ax,format='%.2f',shrink=0.8)

#         # plt.imshow(align_*1e4,interpolation='bilinear',origin='lower',cmap='magma',norm=LogNorm())

        