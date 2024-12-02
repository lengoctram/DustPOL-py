import numpy as np
import matplotlib.pyplot as plt
import pwlf, os
import importlib
from scipy.interpolate import interp1d
from scipy import integrate
from scipy import interpolate
from sympy import Symbol
from sympy.utilities import lambdify
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit
from astropy.io import ascii
from collections import OrderedDict
import DustPOL_class, DustPOL_io
from DustPOL_io import dust_type, f_max, U, alpha, output
# plt.close('all')
ls                  = OrderedDict(
                                 [
                                  ('solid',               (0, ())),
                                  ('dashed',              (0, (5, 5))),
                                  # ('solid',               (0, ())),
                                  ('dashdotted',          (0, (5, 4, 1, 6))),
                                  ('dotted',              (0, (1, 5))),

                                  ('loosely dashed',      (0, (5, 15))),
                                  ('densely dashed',      (0, (5, 1))),
                                  ('loosely dotted',      (0, (1, 10))),
                                  ('densely dotted',      (0, (1, 1))), 
                                  ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),

                                  ('loosely dashdotted',  (0, (3, 10, 1, 10))),
                                  ('densely dashdotted',  (0, (3, 1, 1, 1))),

                                  ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
                                  ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
keys = list(ls.keys())

args = DustPOL_class.DustPOL()
# path='./output/starless/astrodust/U=%.2f_alpha=%.4f/Av_fixed_amax/'%(U,alpha)
# dust_type='astrodust'

path='./output/BHR121/astrodust/U=%.2f_alpha=%.4f/Av_fixed_amax/'%(U,alpha)
dust_type='astrodust'
print('we are working at:', path)
# path='/Users/lengoctram/Documents/postdoc/Leiden/research/BHR121/model/output/astrodust/U=5.00_alpha=3.0000_beta=-3.5/Av_fixed_amax/'
# dust_type='astrodust'

def PiecewiseLineFit(xdata,ydata,fitto='temp',nlines=2,force_break=False, x0=None, get_func=False, sigma=None):
    x = Symbol('x')
    def get_symbolic_eqn(pwlf_, segment_number):
        if pwlf_.degree < 1:
            raise ValueError('Degree must be at least 1')
        if segment_number < 1 or segment_number > pwlf_.n_segments:
            raise ValueError('segment_number not possible')
        # assemble degree = 1 first
        get_beta=[];get_se=[]
        for line in range(segment_number):
            if line == 0:
                my_eqn = pwlf_.beta[0] + (pwlf_.beta[1])*(x-pwlf_.fit_breaks[0])
                get_beta.append(pwlf_.beta[0])
                get_beta.append(pwlf_.beta[1])
            else:
                my_eqn += (pwlf_.beta[line+1])*(x-pwlf_.fit_breaks[line])
                get_beta.append(pwlf_.beta[line+1])
        # assemble all other degrees
        #if pwlf_.degree > 1:
        #    for k in range(2, pwlf_.degree + 1):
        #        for line in range(segment_number):
        #            beta_index = pwlf_.n_segments*(k-1) + line + 1
        #            my_eqn += (pwlf_.beta[beta_index])*(x-pwlf_.fit_breaks[line])**k
        return my_eqn.simplify(), get_beta
 
    if (force_break):
        xx=np.array([min(xdata),x0,max(xdata)])
        my_pwlf_2 = pwlf.PiecewiseLinFit(xdata, ydata)
        res2 = my_pwlf_2.fit_with_breaks(xx)
    else:
        try:
            if sigma==None: my_pwlf_2 = pwlf.PiecewiseLinFit(xdata, ydata)
            else: my_pwlf_2 = pwlf.PiecewiseLinFit(xdata, ydata, weights=sigma)
        except:
            if sigma.any()==None: my_pwlf_2 = pwlf.PiecewiseLinFit(xdata, ydata)
            else: my_pwlf_2 = pwlf.PiecewiseLinFit(xdata, ydata, weights=sigma)
        res2  = my_pwlf_2.fitfast(nlines,pop=50)
    eqn_list = []
    f_list = []
    beta_list=[]
    print ("--------------Fitting information----------------")
    for i in range(my_pwlf_2.n_segments):
        eqns, betas = get_symbolic_eqn(my_pwlf_2, i + 1)
        eqn_list.append(eqns)
        print('Equation number: ', i + 1)
        print(eqn_list[-1])
        f_list.append(lambdify(x, eqn_list[-1]))
    if (fitto=='temp'):
        slopee=betas[2]+betas[1]
        print('Decreased slope = ', slopee)
    elif (fitto=='NH'):
        slopee=betas[1]
        print('Increased slope = ', slopee)
    beta_list.append(betas[1])
    beta_list.append(betas[2]+betas[1])
    xHat = np.linspace(min(xdata), max(xdata), num=10000)
    yHat = my_pwlf_2.predict(xHat)

    print('slopes=',my_pwlf_2.slopes,my_pwlf_2.n_segments)
    p = my_pwlf_2.p_values(method='non-linear', step_size=1e-4)
    se = my_pwlf_2.se  # standard errors
    parameters = np.concatenate((my_pwlf_2.beta,
                                 my_pwlf_2.fit_breaks[1:-1]))

    header = ['Parameter type', 'Parameter value', 'Standard error', 't',
              'P > np.abs(t) (p-value)']
    print(*header, sep=' | ')
    values = np.zeros((parameters.size, 5), dtype=np.object_)
    values[:, 1] = np.around(parameters, decimals=3)
    values[:, 2] = np.around(se, decimals=3)
    values[:, 3] = np.around(parameters / se, decimals=3)
    values[:, 4] = np.around(p, decimals=3)

    for i, row in enumerate(values):
        if i < my_pwlf_2.beta.size:
            row[0] = 'Beta'
            print(*row, sep=' | ')
        else:
            row[0] = 'Breakpoint'
            print(*row, sep=' | ')

    print('se=',se)
    if (force_break):
        x_break=x0
        e_break=0
    else:
        x_break=res2
        e_break=se[-1]
    if (get_func):
        return xHat, yHat, x_break, slopee, res2, f_list, beta_list,standard_error
    else:
        return xHat, yHat, x_break, e_break, slopee, res2

def get_Av(filename):
	##get Av_array
	f=open(filename,'r')
	lines=f.readlines()
	f.close()
	Av_str = lines[5].split(',')
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
	return Av_

def plot_pl(av_range,amax=1.0,pol='abs',color='k',ax=None):
	if ax is None:
		fig,ax=plt.subplots(figsize=(10,6))
	if dust_type=='astrodust' or dust_type=='Astrodust':
		filename   = path+'p_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'
	else:
		if dust_type=='sil':
			filename   = path+'p_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'
		elif dust_type=='mix':
			filename   = path+'p_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'

	Av_=get_Av(filename)

	##get column's name
	f=open(filename,'r')
	lines=f.readlines()
	f.close()
	names = lines[7].split()

	##get the data
	# data=ascii.read(filename,header_start=7, include_names=names)
	data_=np.loadtxt(filename,skiprows=8)

	##wavelength
	#w=data['w'].data
	w=data_[:,0]

	for i,iav in enumerate(av_range):
		if iav>Av_.max() or iav<Av_.min():
			raise IOError('Value of of input Av=%.2f is outof the boundary!'%iav)
			continue

		##Interpolate the computed data over Av_ and w:
		##Follow: https://stackoverflow.com/questions/39332053/using-scipy-interpolate-interpn-to-interpolate-a-n-dimensional-array
		## ---> get the value of p% at a costumized Av_ 
		interp_x=iav
		interp_y=w
		interp_mesh = np.array(np.meshgrid(interp_x, interp_y))
		interp_points = np.rollaxis(interp_mesh, 0, 3).reshape((len(w), 2))

		if pol=='abs':
			interp_arr = interpolate.interpn((Av_,w), data_[:,1:].T, interp_points)
			ax.semilogx(w,interp_arr/iav,color=color,ls=ls[keys[i]],label='$\\sf A_{V}=%.0f$'%iav)
		elif pol=='emi':
			correct_data=data_[:,1:][:,::2]
			interp_arr = interpolate.interpn((Av_,w), correct_data.T, interp_points)
			ax.semilogx(w,interp_arr,color=color,ls=ls[keys[i]],label='$\\sf A_{V}=%.0f$'%iav)

	# if float(alpha<1):
	#     plt.text(0.8, 0.73, '$\\sf prolate\\, grain: s=1/3$', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	# else:
	#     plt.text(0.8, 0.73, '$\\sf oblate\\, shape: s=2$', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	# plt.text(0.8, 0.7, '$\\sf f_{max}=%.1f$'%(f_max), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	ax.set_xlabel('$\\sf Wavelength\\, (\\mu m)$')
	if pol=='abs':
		ax.text(0.8, 0.9, '$\\sf aligned\\, grain: %s$'%dust_type, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.text(0.8, 0.8, '$\\sf a_{max}=%.2f\\, \\mu m$'%amax, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.set_ylabel('$\\sf p_{ext}/A_{V}\\,(\\%/mag.)$')
		ax.legend(loc='lower right',bbox_to_anchor=(0.93,0.4),fontsize=18)
	elif pol=='emi':
		ax.text(0.2, 0.9, '$\\sf aligned\\, grain: %s$'%dust_type, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.text(0.2, 0.8, '$\\sf a_{max}=%.2f\\, \\mu m$'%amax, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.set_ylabel('$\\sf p_{em}\\,(\\%)$')
		ax.legend(loc='lower left',bbox_to_anchor=(0.07,0.4),fontsize=18)

	# plt.ylim([1e-26,5e-21])
	ax.get_xaxis().set_major_formatter(ScalarFormatter())
	#ax.get_yaxis().set_minor_formatter(ScalarFormatter())
	ax.get_xaxis().set_major_formatter(ScalarFormatter())
	if pol=='abs':
		ax.set_xlim([0.03,1e3])
	elif pol=='emi':
		ax.set_xlim([10,3000])
	# plt.savefig('Plambda_abs_amax=%.2f.pdf'%(amax))
	plt.show()

def plot_pav(amax_range,lam,color='k',pol='abs',ax=None,show_break=False):
	if ax is None:
		fig,ax=plt.subplots(figsize=(7.5,9))
	# save={}
	# filesave='models.dat'
	for i,amax in enumerate(amax_range):
		if dust_type=='astrodust' or dust_type=='Astrodust':
			filename   = path+'p_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'
		else:
			if dust_type=='sil':
				filename   = path+'p_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'
			elif dust_type=='mix':
				filename   = path+'pmix_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'

		data=np.loadtxt(filename,skiprows=8)
		Av_=get_Av(filename); new_Av_=np.linspace(Av_.min(),Av_.max(),100)
		w  =data[:,0]
		x,y=np.shape(data)
		data_=[];I_=[]
		if pol=='abs':
			for j in range(1,y):
				data_.append(interp1d(w*1e-4,data[:,j],axis=0)(lam*1e-4))#(0.65e-4))

		elif pol=='emi':
			for j in range(1,y,2):
				data_.append(interp1d(w*1e-4,data[:,j],axis=0)(lam*1e-4))#(0.65e-4))

			# for j in range(1,y):
			# 	data_.append(interp1d(w*1e-4,data[:,j],axis=0)(lam*1e-4))#(0.65e-4))
			# I_=data_[1::2]
			# data_=data_[::2]

		data_=np.array(data_); print(len(Av_),len(data_))
		f_ip = interpolate.interp1d(Av_,data_,kind='cubic')
		new_data_=f_ip(new_Av_)#*np.sin(30.*np.pi/180)*np.sin(30.*np.pi/180)

		if pol=='abs':
			#plt.loglog(Av_,data_/Av_,color='k',ls=ls[keys[i]],label='$\\sf a_{max}=%.1f\\, \\mu m$'%(amax))
			ax.loglog(new_Av_,new_data_/new_Av_,color=color,ls=ls[keys[i]],label='$\\sf a_{max}=%.2f\\, \\mu m$'%(amax))
			x=np.log10(new_Av_)
			# x=np.log10(Av_num[idd])
			y=np.log10(new_data_/new_Av_)
			f=interp1d(new_Av_,new_data_/new_Av_)
			# save['Av(amax=%.2f)'%(amax)]=new_Av_
			# save['p/Av(amax=%.2f)'%(amax)]=new_data_/new_Av_
			# output(args,filesave,save).file_save()
			if (show_break):
				xHAT, yHAT, xbreak, ebreak, slope, resdust = PiecewiseLineFit(x, y,nlines=2)
				x0=10**(xbreak[1])
				y0=f(x0)
				ax.plot(x0,y0,'o',color=color)
				ax.vlines(x=x0,ymin=y0,ymax=100,color='gray',ls='-.',lw=1.0)

		elif pol=='emi':
			ax.loglog(Av_,data_,color=color,ls=ls[keys[i]],label='$\\sf a_{max}=%.1f\\, \\mu m$'%(amax))
			# plt.loglog(new_Av_,new_data_,color='k',ls=ls[keys[i]],label='$\\sf a_{max}=%.1f\\, \\mu m$'%(amax))

			x=np.log10(Av_)#[Av_>=1.5])
			y=np.log10(data_)#[Av_>=1.5])
			f=interp1d(Av_,data_)

			if (show_break):
				xHAT, yHAT, xbreak, ebreak, slope, resdust = PiecewiseLineFit(x, y,nlines=3)
				x0=10**(xbreak[2])
				y0=f(x0)
				ax.plot(x0,y0,'o',color=color)
				ax.vlines(x=x0,ymin=y0,ymax=100,color='gray',ls='-.',lw=1.0)
				# ax.plot(10**(xHAT),10**(yHAT),'r-')
	if pol=='abs':
		ax.text(0.7, 0.9, '$\\sf \\lambda= %.2f\\, \\mu m$'%(lam), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.text(0.7, 0.83, '$\\sf aligned\\, grain: %s$'%dust_type, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		# ax.set_xticks([10.0],['10'])
		ax.legend(loc='lower left',bbox_to_anchor=(0.05,0.01),fontsize=18)
		ax.set_xlabel('$\\sf A_{V}\\, (mag.)$')
		ax.set_ylabel('$\\sf p_{ext}/A_{V}\\,(\\%/mag.)$')
		# ax.set_xlim([1,20])
		# ax.set_ylim([0.4,7])
		# ax.set_xticks([10.0])
		# ax.set_xticklabels(['10'])
		ax.set_xticks([1.00])
		ax.set_xticklabels(['1.0'])

	elif pol=='emi':
		ax.text(0.7, 0.9, '$\\sf \\lambda= %.0f\\, \\mu m$'%(lam), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.text(0.7, 0.83, '$\\sf aligned\\, grain: %s$'%dust_type, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.set_xticks([10.0])
		ax.set_xticklabels(['10'])
		ax.legend(loc='lower left',bbox_to_anchor=(0.05,0.01),fontsize=18)
		ax.set_xlabel('$\\sf A_{V}\\, (mag.)$')
		ax.set_ylabel('$\\sf p_{em}\\,(\\%)$')
		ax.set_xlim([5,60])
		# ax.set_ylim([0.01,3])
	ax.get_xaxis().set_major_formatter(ScalarFormatter())
	ax.get_xaxis().set_minor_formatter(ScalarFormatter())
	ax.get_yaxis().set_major_formatter(ScalarFormatter())
	ax.get_yaxis().set_minor_formatter(ScalarFormatter())

	plt.show()
	return new_Av_,new_data_/new_Av_
##set exponential format for figure's axes
# current_values = plt.gca().get_yticks()
# plt.gca().set_yticklabels(['{:,.1e}'.format(x) for x in current_values])


def plot_pI(amax_range,lam,pol='emi',ax=None):
	if ax is None:
		fig,ax=plt.subplots(figsize=(7.5,9))
	for i,amax in enumerate(amax_range):
		if dust_type=='astrodust' or dust_type=='Astrodust':
			filename   = path+'p_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'
		else:
			if dust_type=='sil':
				filename   = path+'p_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'
			elif dust_type=='mix':
				filename   = path+'mix_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'

		data=np.loadtxt(filename,skiprows=8)
		w  =data[:,0]
		x,y=np.shape(data)
		data_=[];I_=[]
		
		for j in range(1,y,2):
			data_.append(interp1d(w*1e-4,data[:,j],axis=0)(lam*1e-4))#(0.65e-4))
		for j in range(2,y,2):
			I_.append(interp1d(w*1e-4,data[:,j],axis=0)(lam*1e-4))#(0.65e-4))

		data_=np.array(data_); I_=np.array(I_)
		new_I_=np.linspace(I_.min(),I_.max(),100)
		f_ip = interp1d(I_,data_,kind='cubic')
		new_data_=f_ip(new_I_)
		ax.loglog(I_/I_.max(),data_,color='k',ls=ls[keys[i]],label='$\\sf a_{max}=%.1f\\, \\mu m$'%(amax))
		# plt.loglog(new_I_/new_I_.max(),new_data_,color='k',ls=ls[keys[i]],label='$\\sf a_{max}=%.1f\\, \\mu m$'%(amax))

	# I_loss=np.linspace(0.2,1,20)
	# p_loss=pow(I_loss,-1)
	# plt.plot(I_loss[p_loss<=10],1.5*p_loss[p_loss<=10],color='gray')
	# fp_loss = interp1d(I_loss,p_loss)
	# plt.text(0.35, fp_loss(0.35)+2.0, '$\\sf I^{-1}$', color='gray', horizontalalignment='center', verticalalignment='center',rotation=-55)

	ax.text(0.7, 0.9, '$\\sf \\lambda= %.0f\\, \\mu m$'%(lam), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	ax.text(0.7, 0.83, '$\\sf aligned\\, grain: %s$'%dust_type, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	#plt.xticks([10.0],['10'])
	ax.legend(loc='lower left',bbox_to_anchor=(0.05,0.01),fontsize=18)
	ax.set_xlabel('$\\sf I/max(I)$')
	ax.set_ylabel('$\\sf p_{em}\\,(\\%)$')
	#plt.xlim([5,60])
	# plt.ylim([0.01,3])
	ax.get_xaxis().set_major_formatter(ScalarFormatter())
	#ax.get_xaxis().set_minor_formatter(ScalarFormatter())
	ax.get_yaxis().set_major_formatter(ScalarFormatter())
	ax.get_yaxis().set_minor_formatter(ScalarFormatter())

##set exponential format for figure's axes
# current_values = plt.gca().get_yticks()
# plt.gca().set_yticklabels(['{:,.1e}'.format(x) for x in current_values])

def plot_lamav(av_range,amax_range,color='k'):
	fig,ax=plt.subplots(figsize=(7.5,9))	
	for j,amax in enumerate(amax_range):
		if dust_type=='astrodust' or dust_type=='Astrodust':
			filename   = path+'p_amax=%.2f'%(amax)+'_abs.dat'
		else:
			if dust_type=='sil':
				filename   = path+'p_amax=%.2f'%(amax)+'_abs.dat'
			elif dust_type=='mix':
				filename   = path+'p_amax=%.2f'%(amax)+'_abs.dat'

		Av_=get_Av(filename)

		##get column's name
		f=open(filename,'r')
		lines=f.readlines()
		f.close()
		names = lines[7].split()

		##get the data
		# data=ascii.read(filename,header_start=7, include_names=names)
		data_=np.loadtxt(filename,skiprows=8)

		##wavelength
		#w=data['w'].data
		w=data_[:,0]

		lam_max=np.zeros(len(av_range))
		for i,iav in enumerate(av_range):
			if iav>Av_.max() or iav<Av_.min():
				raise IOError('Value of of input Av=%.2f is outof the boundary!'%iav)
				continue

			##Interpolate the computed data over Av_ and w:
			##Follow: https://stackoverflow.com/questions/39332053/using-scipy-interpolate-interpn-to-interpolate-a-n-dimensional-array
			## ---> get the value of p% at a costumized Av_ 
			interp_x=iav
			interp_y=w
			interp_mesh = np.array(np.meshgrid(interp_x, interp_y))
			interp_points = np.rollaxis(interp_mesh, 0, 3).reshape((len(w), 2))

			interp_arr = interpolate.interpn((Av_,w), data_[:,1:].T, interp_points)

			##Finding lambda max
			idl = np.argmax(interp_arr)
			lam_max[i]=w[idl]

		ax.plot(av_range,lam_max,color=color,ls=ls[keys[j]],label='$\\sf a_{\\rm max}=%.2f\\,\\mu m$'%amax)

	ax.set_xlabel('$\\sf A_{V}\\, (mag.)$')
	ax.text(0.27, 0.9, '$\\sf aligned\\, grain: %s$'%dust_type, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	# ax.text(0.25, 0.85, '$\\sf a_{max}=%.2f\\, \\mu m$'%amax, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	ax.set_ylabel('$\\sf \\lambda_{max}\\,(\\mu m)$')
	ax.legend(loc='lower right',bbox_to_anchor=(0.93,0.4),fontsize=18)

	# plt.ylim([1e-26,5e-21])
	ax.get_xaxis().set_major_formatter(ScalarFormatter())
	#ax.get_yaxis().set_minor_formatter(ScalarFormatter())
	ax.get_xaxis().set_major_formatter(ScalarFormatter())
	# if pol=='abs':
	# 	ax.set_xlim([0.03,1e3])
	# elif pol=='emi':
	# 	ax.set_xlim([10,3000])
	# # plt.savefig('Plambda_abs_amax=%.2f.pdf'%(amax))
	plt.show()	

def plot_pl_s(s_range,av_range,amax=1.0,pol='abs',color=['black','gray']):
	fig,ax=plt.subplots(figsize=(10,6))

	legend_=True
	llines=[]
	slegend=[]
	for j,s_ in enumerate(s_range):
		path='./output/starless/astrodust/U=3.00_alpha=%.4f/Av_fixed_amax/'%(s_)

		if dust_type=='astrodust' or dust_type=='Astrodust':
			filename   = path+'p_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'
		else:
			if dust_type=='sil':
				filename   = path+'p_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'
			elif dust_type=='mix':
				filename   = path+'p_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'

		Av_=get_Av(filename)

		##get column's name
		f=open(filename,'r')
		lines=f.readlines()
		f.close()
		names = lines[7].split()

		##get the data
		# data=ascii.read(filename,header_start=7, include_names=names)
		data_=np.loadtxt(filename,skiprows=8)

		##wavelength
		#w=data['w'].data
		w=data_[:,0]

		for i,iav in enumerate(av_range):
			if iav>Av_.max() or iav<Av_.min():
				raise IOError('Value of of input Av=%.2f is outof the boundary!'%iav)
				continue

			if (legend_):
				label='$\\sf A_{V}=%.0f$'%iav
			else:
				label=None
			##Interpolate the computed data over Av_ and w:
			##Follow: https://stackoverflow.com/questions/39332053/using-scipy-interpolate-interpn-to-interpolate-a-n-dimensional-array
			## ---> get the value of p% at a costumized Av_ 
			interp_x=iav
			interp_y=w
			interp_mesh = np.array(np.meshgrid(interp_x, interp_y))
			interp_points = np.rollaxis(interp_mesh, 0, 3).reshape((len(w), 2))

			if pol=='abs':
				interp_arr = interpolate.interpn((Av_,w), data_[:,1:].T, interp_points)
				ax.semilogx(w,interp_arr/iav,color=color[j],ls=ls[keys[i]],label=label)
			elif pol=='emi':
				correct_data=data_[:,1:][:,::2]
				interp_arr = interpolate.interpn((Av_,w), correct_data.T, interp_points)
				ax.semilogx(w,interp_arr,color=color[j],ls=ls[keys[i]],label=label)
		legend_=False

		##secondary legend
		line = ax.plot(Av_,-2*np.ones(len(Av_)),color=color[j])
		llines.append(line)
		slegend.append('$\\sf axial\\,ratio\\,:\\,%.1f$'%(s_))

	print(llines,np.shape(llines))
	from matplotlib.legend import Legend
	##leg = Legend(ax, lines, ['$\\sf B-RAT$','$\\sf k-RAT$','$\\sf disruption\\newline(S_{max,6}\\rightarrow S_{max,8})$'], \
	##            loc='upper left', frameon=True,facecolor='black',edgecolor='black',labelcolor='white')
	llines = np.array(llines).reshape(len(s_range),)
	leg = Legend(ax, llines, slegend, loc='lower right', frameon=False, fontsize=18)
	ax.add_artist(leg)

	# if float(alpha<1):
	#     plt.text(0.8, 0.73, '$\\sf prolate\\, grain: s=1/3$', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	# else:
	#     plt.text(0.8, 0.73, '$\\sf oblate\\, shape: s=2$', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	# plt.text(0.8, 0.7, '$\\sf f_{max}=%.1f$'%(f_max), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
	ax.set_xlabel('$\\sf Wavelength\\, (\\mu m)$')
	if pol=='abs':
		ax.text(0.8, 0.9, '$\\sf aligned\\, grain: %s$'%dust_type, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.text(0.8, 0.8, '$\\sf a_{max}=%.2f\\, \\mu m$'%amax, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.set_ylabel('$\\sf p/A_{V}\\,(\\%/mag.)$')
		ax.legend(loc='lower right',bbox_to_anchor=(0.93,0.4),fontsize=18)
	elif pol=='emi':
		ax.text(0.2, 0.9, '$\\sf aligned\\, grain: %s$'%dust_type, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.text(0.2, 0.8, '$\\sf a_{max}=%.2f\\, \\mu m$'%amax, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.set_ylabel('$\\sf p\\,(\\%)$')
		ax.legend(loc='lower left',bbox_to_anchor=(0.07,0.4),fontsize=18)

	# plt.ylim([1e-26,5e-21])
	ax.get_xaxis().set_major_formatter(ScalarFormatter())
	#ax.get_yaxis().set_minor_formatter(ScalarFormatter())
	ax.get_xaxis().set_major_formatter(ScalarFormatter())
	if pol=='abs':
		ax.set_xlim([0.03,1e3])
	elif pol=='emi':
		ax.set_xlim([10,3000])
	# plt.savefig('Plambda_abs_amax=%.2f.pdf'%(amax))
	plt.show()

def plot_avbreak(amax_range,lam_range=None,color='k',pol='abs',ax=None):
	if ax is None:
		fig,ax=plt.subplots(figsize=(7.5,9))
	if lam_range is None:
		lam_range=[0.55,0.65,1.22,1.65,2.19]
		bands=['V','R','J','H','K']

	for k,lam in enumerate(lam_range):
		av_break=[]
		band=bands[k]		
		for i,amax in enumerate(amax_range):
			if dust_type=='astrodust' or dust_type=='Astrodust':
				filename   = path+'p_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'
			else:
				if dust_type=='sil':
					filename   = path+'p_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'
				elif dust_type=='mix':
					filename   = path+'pmix_amax=%.2f'%(amax)+'_'+str(pol)+'.dat'

			data=np.loadtxt(filename,skiprows=8)
			Av_=get_Av(filename); new_Av_=np.linspace(Av_.min(),Av_.max(),100)
			w  =data[:,0]
			x,y=np.shape(data)
			data_=[];I_=[]
			if pol=='abs':
				for j in range(1,y):
					data_.append(interp1d(w*1e-4,data[:,j],axis=0)(lam*1e-4))#(0.65e-4))

			elif pol=='emi':
				for j in range(1,y,2):
					data_.append(interp1d(w*1e-4,data[:,j],axis=0)(lam*1e-4))#(0.65e-4))

				# for j in range(1,y):
				# 	data_.append(interp1d(w*1e-4,data[:,j],axis=0)(lam*1e-4))#(0.65e-4))
				# I_=data_[1::2]
				# data_=data_[::2]

			data_=np.array(data_); print(len(Av_),len(data_))
			f_ip = interpolate.interp1d(Av_,data_,kind='cubic')
			new_data_=f_ip(new_Av_)#*np.sin(30.*np.pi/180)*np.sin(30.*np.pi/180)

			if pol=='abs':
				x=np.log10(new_Av_)
				# x=np.log10(Av_num[idd])
				y=np.log10(new_data_/new_Av_)
				f=interp1d(new_Av_,new_data_/new_Av_)

				xHAT, yHAT, xbreak, ebreak, slope, resdust = PiecewiseLineFit(x, y,nlines=2)
				x0=10**(xbreak[1])
				y0=f(x0)
				# ax.plot(x0,y0,'o',color=color)
				# ax.vlines(x=x0,ymin=y0,ymax=100,color='gray',ls='-.',lw=1.0)
				av_break.append(x0)

			elif pol=='emi':
				x=np.log10(Av_)#[Av_>=1.5])
				y=np.log10(data_)#[Av_>=1.5])
				f=interp1d(Av_,data_)

				xHAT, yHAT, xbreak, ebreak, slope, resdust = PiecewiseLineFit(x, y,nlines=3)
				x0=10**(xbreak[2])
				y0=f(x0)
				# ax.plot(x0,y0,'o',color=color)
				# ax.vlines(x=x0,ymin=y0,ymax=100,color='gray',ls='-.',lw=1.0)
				av_break.append(x0)

		ax.plot(amax_range,av_break,color=color,ls=ls[keys[k]],label='$\\sf \\lambda= %.2f\\, \\mu m\\, (%s-band)$'%(lam,band))

	if pol=='abs':
		# ax.text(0.7, 0.9, '$\\sf \\lambda= %.2f\\, \\mu m$'%(lam), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.text(0.7, 0.1, '$\\sf aligned\\, grain: %s$'%dust_type, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		# ax.set_xticks([10.0],['10'])
		ax.legend(loc='upper left',bbox_to_anchor=(0.05,0.98),fontsize=15)
		ax.set_xlabel('$\\sf a_{max}\\, (\\mu m)$')
		ax.set_ylabel('$\\sf A^{loss}_{V}\\,(mag.)$')
		# ax.set_xlim([1,20])
		# ax.set_ylim([0.4,7])
		# ax.set_xticks([10.0])
		# ax.set_xticklabels(['10'])
	elif pol=='emi':
		# ax.text(0.7, 0.9, '$\\sf \\lambda= %.0f\\, \\mu m$'%(lam), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		ax.text(0.7, 0.1, '$\\sf aligned\\, grain: %s$'%dust_type, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
		# ax.set_xticks([10.0])
		# ax.set_xticklabels(['10'])
		ax.legend(loc='upper left',bbox_to_anchor=(0.05,0.98),fontsize=15)
		ax.set_xlabel('$\\sf a_{max}\\, (\\mu m)$')
		ax.set_ylabel('$\\sf A^{loss}_{V}\\,(mag.)$')
		# ax.set_xlim([5,60])
		# ax.set_ylim([0.01,3])
	# ax.get_xaxis().set_major_formatter(ScalarFormatter())
	# ax.get_xaxis().set_minor_formatter(ScalarFormatter())
	# ax.get_yaxis().set_major_formatter(ScalarFormatter())
	# ax.get_yaxis().set_minor_formatter(ScalarFormatter())

	plt.show()

plt.show()

