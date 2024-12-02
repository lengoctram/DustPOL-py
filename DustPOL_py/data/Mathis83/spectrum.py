from execfile import *
from read import *

c = 3*10**10 #cm/s
Av = [0,3,5,10,20,50]

def uISRF():
    lamb = readD(0)[0,:]
    uISRF = readD(0)[1,:]
    
    u_ISRF1 = np.trapz(uISRF,x=lamb)
    
    lmax = where(abs(lamb-20.e-4) == min(abs(lamb-20.e-4)))
    lrange = lamb[0:lmax[0][0]]
    urange = uISRF[0:lmax[0][0]]
    
    u_ISRF2 = np.trapz(urange,x=lrange)
    return u_ISRF1,u_ISRF2

def LambU(iv,u_ISRF1,u_ISRF2):
# reproduce spectrum of Mathis83'
    lamb = readD(iv)[0,:]
    urad = readD(iv)[1,:]
    rad = urad * lamb * c
# radiation factor
    u_rad1 = np.trapz(urad,x=lamb)
    lambA1 = np.trapz(urad*lamb,x=lamb)/u_rad1 * 1.e4
    
    U1 = u_rad1 / u_ISRF1
# radiation factor in a range of 0.1 - 20 microns where radiation from star is absorbed and polarized by small grains
    lmax = where(abs(lamb-20.e-4) == min(abs(lamb-20.e-4)))
    lrange = lamb[0:lmax[0][0]]
    urange = urad[0:lmax[0][0]]
    
    u_rad2 = np.trapz(urange,x=lrange)
    lambA2 = np.trapz(urange*lrange,x=lrange)/u_rad2 * 1.e4
    #print (Av[iv],lmax)
    
    U2 = u_rad2 / u_ISRF2
    plt.figure(1)
    plt.plot(lamb*1.e4,rad,label=r'Av='+str(Av[iv]))
    return U1, lambA1, U2, lambA2

u_ISRF1 = uISRF()[0]
u_ISRF2 = uISRF()[1]
print(u_ISRF1,u_ISRF2)
for iv in range(0,len(Av)):
    UINDEX1 = LambU(iv,u_ISRF1,u_ISRF2)[0]
    lambA1 = LambU(iv,u_ISRF1,u_ISRF2)[1]
    
    UINDEX2 = LambU(iv,u_ISRF1,u_ISRF2)[2]
    lambA2 = LambU(iv,u_ISRF1,u_ISRF2)[3]
    print(Av[iv],'%4.2f'%(UINDEX1),lambA1,'%4.2f'%(UINDEX2),lambA2)

plt.figure(1)
plt.axis([0.1,1.e3,1.e-5,0.1])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\lambda(\mu$m)')
plt.ylabel(r'4$\pi$J$_{\lambda}\lambda$(erg$\,$cm$^{-2}\,$s$^{-1}$)')
#plt.ylabel(r'4$\piJ_{\lambda}\lambda$(erg\,cm$^{-2}$s$^{-1}$')
plt.show()
