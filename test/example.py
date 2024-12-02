import numpy as np
import os
import matplotlib.pyplot as plt
from DustPOL_py.DustPOL_class import DustPOL
from scipy.signal import savgol_filter
from IPython.display import clear_output
from astropy import constants
pc=constants.pc.cgs.value
# os.system("%config InteractiveShell.cache_size = 0")

#starlight polarisation
exe = DustPOL('input_template.dustpol') 
# w,pext,_= exe.cal_pol_abs()
# ext_curve = exe.extinction()

# fig1, ax1 = plt.subplots(figsize=(12, 5))
# ax1.set_xlabel('$\\rm wavelength\\,(\\mu m)$')
# ax1.set_ylabel('$\\rm p_{ext}/N_{H}\\,(\\%/cm^{-2})$')
# ax1.set_title('$\\rm Starlight\\,Polarization$',pad=20)
# ax11=ax1.twinx()
# ax11.set_ylabel('$\\rm A_{\\lambda}/N_{\\rm H}$')

# # ax.semilogx(w*1e4,p)
# # ax.semilogx(w*1e4,ext_curve,'k-')
# # ax.set_xlabel('wavelength ($\\mu$m)')
# # ax.set_ylabel(r'$P_{\rm ext}(\%)/N_{\rm gas}$')


# pext = savgol_filter(pext,20,2) # smooth pext (for visualization) -- not physically affected
# ax1.semilogx(w * 1e4, pext)#, label=f'U={U_rad:.1f} -- n$_{{\\rm H}}$={n_gas:.1e} -- f$_{{\\rm max}}$={f_max:.2f}')
# ax11.loglog(w * 1e4, ext_curve,color='k',ls='--')
                 
# ax11.loglog(w*1e4,np.ones(len(w)),color='k',ls='-',label='$\\rm pol.\\,spectrum$')
# ax11.loglog(w*1e4,np.ones(len(w)),color='k',ls='--',label='$\\rm Extinction\\, curve$')
                        
# ax11.legend(bbox_to_anchor=(0.95,0.95),frameon=False)
# ax11.set_ylim([1e-23,1e-20])


# #thermal dust polarisation
# w,I,p=exe.cal_pol_emi()
# fig,ax=plt.subplots(figsize=(9,3))
# ax.semilogx(w*1e4,p[0])
# ax.set_xlabel('wavelength ($\\mu$m)')
# ax.set_ylabel(r'$P_{\rm emi}(\%)$')

# i=1
# for los in np.array([0.0,0.1,0.3]):
# 	exe.isoCloud_los(los*pc,progress=True,save_output=True,filename_output=f"pol_r0={los:.2f}pc")
# 	i=i+1

def get_Av(filename):
	##get Av_array
	f=open(filename,'r')
	lines=f.readlines()
	f.close()
	Av_str = lines[6].split(',')
	Av_ls=[]
	for i in range(len(Av_str)):
		try:
			Av_ls.append(eval(Av_str[i]))
		except:
			redo=Av_str[i].split()
			for j in range(len(redo)):
				try:
					Av_ls.append(eval(redo[j]))
				except:
					continue
	Av_=np.array(Av_ls)
	Av_=np.array(list(dict.fromkeys(Av_))) ##de-duplicated values
	if len(Av_)==1:
		return float(Av_)
	else:
		return Av_

def plot_pl(filenames,composition='single',ax=None,**plot_kwargs):
	if ax is None:
		fig,ax=plt.subplots(figsize=(10,6))

	# if isinstance(filenames,list) or isinstance(filenames,np.array):
	if isinstance(filenames,str):
		filenames = [filenames]

	for i,filename in enumerate(filenames):
		Av_=get_Av(filename)

		##get column's name
		f=open(filename,'r')
		lines=f.readlines()
		f.close()
		names = lines[7].split()

		##get the data
		# data=ascii.read(filename,header_start=7, include_names=names)
		data_=np.loadtxt(filename,skiprows=9)

		##wavelength
		#w=data['w'].data
		w=data_[:,0]
		if composition=='single':
			p=data_[:,1]
		else:
			p=data_[:,2]

		if 'abs' in filename:
			ax.semilogx(w,p/Av_,**plot_kwargs,label='$\\rm A_{V}=%.2f mag.$'%Av_)
		elif 'emi' in filename:
			ax.semilogx(w,p,**plot_kwargs,label='$\\rm A_{V}=%.2f mag.$'%Av_)
		else:
			raise IOError('')
	ax.set_xlabel('$\\rm Wavelength\\, (\\mu m)$')
	if 'abs' in filenames[0]:	
		ax.set_ylabel('$\\rm p_{ext}/A_{V}\\,(\\%/mag.)$')
		ax.text(0.8, 0.9, '$\\rm aligned\\, grain: %s$'%(exe.dust_type), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.text(0.8, 0.8, '$\\rm a_{max}=%.2f\\, \\mu m$'%(exe.amax*1e4), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.legend(bbox_to_anchor=(1.0,0.5))
	elif 'emi' in filenames[0]:
		ax.set_ylabel('$\\rm p_{em}\\,(\\%)$')
		ax.text(0.2, 0.9, '$\\rm aligned\\, grain: %s$'%(exe.dust_type), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.text(0.2, 0.8, '$\\rm a_{max}=%.2f\\, \\mu m$'%(exe.amax*1e4), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.legend(bbox_to_anchor=(0.3,0.5))
	else:
		raise IOError('')	
	



	# # # plt.ylim([1e-26,5e-21])
	# # ax.get_xaxis().set_major_formatter(ScalarFormatter())
	# # #ax.get_yaxis().set_minor_formatter(ScalarFormatter())
	# # ax.get_xaxis().set_major_formatter(ScalarFormatter())
	# plt.show()

plt.show()