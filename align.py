from plt_setup import *
from read import *
from common import *
from sys import exit
# ------------------------------------------------------
# [DESCRIPTION] aligned grain size and alignment function
# input:
#	U = radiation field strength
# output:
#	1. a_ali = aligned grain size at a radiation field strength
#	2. f = alignment function at a radiation field strength
# ------------------------------------------------------
"""
Sukhbinder
5 April 2017
Based on:
"""
def _rect_inter_inner(x1,x2):
   n1=x1.shape[0]-1
   n2=x2.shape[0]-1
   X1=c_[x1[:-1],x1[1:]]
   X2=c_[x2[:-1],x2[1:]]
   S1=tile(X1.min(axis=1),(n2,1)).T
   S2=tile(X2.max(axis=1),(n1,1))
   S3=tile(X1.max(axis=1),(n2,1)).T
   S4=tile(X2.min(axis=1),(n1,1))
   return S1,S2,S3,S4

def _rectangle_intersection_(x1,y1,x2,y2):
   S1,S2,S3,S4=_rect_inter_inner(x1,x2)
   S5,S6,S7,S8=_rect_inter_inner(y1,y2)

   C1=np.less_equal(S1,S2)
   C2=np.greater_equal(S3,S4)
   C3=np.less_equal(S5,S6)
   C4=np.greater_equal(S7,S8)

   ii,jj=np.nonzero(C1 & C2 & C3 & C4)
   return ii,jj

def intersection(x1,y1,x2,y2):
   """
   INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
   usage:
      x,y=intersection(x1,y1,x2,y2)
   Example:
   a, b = 1, 2
   phi = np.linspace(3, 10, 100)
   x1 = a*phi - b*np.sin(phi)
   y1 = a - b*np.cos(phi)
   x2=phi
   y2=np.sin(phi)+2
   x,y=intersection(x1,y1,x2,y2)
   plt.plot(x1,y1,c='r')
   plt.plot(x2,y2,c='g')
   plt.plot(x,y,'*k')
   plt.show()
   """
   ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
   n=len(ii)
   
   dxy1=np.diff(c_[x1,y1],axis=0)
   dxy2=np.diff(c_[x2,y2],axis=0)

   T=np.zeros((4,n))
   AA=np.zeros((4,4,n))
   AA[0:2,2,:]=-1
   AA[2:4,3,:]=-1
   AA[0::2,0,:]=dxy1[ii,:].T
   AA[1::2,1,:]=dxy2[jj,:].T

   BB=np.zeros((4,n))
   BB[0,:]=-x1[ii].ravel()
   BB[1,:]=-x2[jj].ravel()
   BB[2,:]=-y1[ii].ravel()
   BB[3,:]=-y2[jj].ravel()

   for i in range(n):
      try:
         T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
      except:
         T[:,i]=np.NaN

   in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

   xy0=T[2:,in_range]
   xy0=xy0.T
   return xy0[:,0],xy0[:,1]

# ================================ a_ali function
def Qgam(a,lamb):
    if a <= lamb/1.8:
        return 2 * (lamb/a)**(-2.7)
    else:
        return 0.4
#
def torq(a,U,gamma,lamb):
    torq_rat = gamma*a**2 * (U*u_ISRF) * lamb * (Qgam(a,lamb)/2.)
    return torq_rat
#

# total damping time
def tdamp(a, U, nH, T_gas):
    
    a_7 = a/1e-7
    FIR    = (60.8/a_7)*pow(U,2/3.)*(20./nH)*(100./T_gas)**0.5
    
    t_gas = 8.74e4 * (a/1e-5) * (rho/3) * (30/nH) * sqrt(100./T_gas)*yr # in sec
    t_damp = t_gas/(1. + FIR)
    
    return t_damp
#

def wrat(a,U,gamma,lamb, nH, T_gas):
    
    #FIR = 9.1e-6 * U**(2/3.) * (30/nH) * sqrt(100/T_gas)/a

    a_7 = a/1e-7
    FIR    = (60.8/a_7)*pow(U,2/3.)*(20./nH)*(100./T_gas)**0.5

    t_gas = 8.74e4 * (a/1e-5) * (rho/3) * (30/nH) * sqrt(100./T_gas)*yr # in sec
    t_damp = t_gas/(1 + FIR)
    Ia = 8*pi*rho*a**5./15.
    w_rat = torq(a,U,gamma,lamb) * t_damp/Ia # [Hz] : years -> seconds
    w_th  = sqrt(K*T_gas/Ia)
    return w_rat,w_th
    
def Aligned_Size(U,n_gas,T_gas):
    a = logspace(log10(amin), log10(amax), 5000) # [cm]: recreate an arbitrary smooth grain-size array
    na = len(a)
    ratio=[]
    for j in range(na):
        #F_IR = (0.91/(a[j]*1e5)) * U**(2/3) * (30./n_gas) *(100./T_gas)**0.5
        #OMEGA_RAT = 200* (30./n_gas) *(100/T_gas) *U *2.4*1e-3/1e-2*(a[j]/1e-5)**3.2/(1+F_IR)
        #F_IR = (0.4/(a[j]*1.e5)) * U**(2./3) * (30./n_gas) *(100./T_gas)**0.5
        #OMEGA_RAT = 4174. *pow(rho/3,0.5) * gamma * (lambA*1e4/0.5)**(-1.7) * (a[j]/1.e-5)**(3.2) *U * (10./n_gas) * (1./(1+F_IR)) * (100./T_gas)
        #print('U=',U)
        #print('check=',abs(OMEGA_RAT))
        #print('ngas=',n_gas)
        WWrat, WWth = wrat(a[j],U,gamma,lambA,n_gas, T_gas)
        ratio.append(WWrat/WWth)
        #print('ratio=',ratio)
        #a_ali=intersection(a,ratio,a,3*ones(na))[0]
        #if abs(ratio - 3.0) < 1e-2:
        #    a_ali = a[j]
        #else:
        #    a_ali = max(a)
    if max(ratio)<3.0:
        log.warning('*** \033[1;5;33m St<3, set a_align == a_max \033[0m')
    try:
        a_ali=intersection(a,array(ratio),a,3*ones(na))[0][0]
    except:
        a_ali=max(a)
    return a_ali

# ==============ALIGNMENT FUNCTION
def f_ali(a,a_ali):
    #log.info('Maximum grain alignment efficiency: f_max=%.2f'%(f_max))
    f_min=1e-3
    #f_max=f_max#0.35#1.
    f = f_min + (f_max-f_min)*(1-np.exp(-(0.5*a/a_ali)**3))
    #f  = 1.0-np.exp(-(0.5*a/a_ali)**3.0)
    return f
