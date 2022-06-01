from plt_setup import *
import os
from astropy import log
# -------------------------------------------------------------------------
# Physical constants
# -------------------------------------------------------------------------
H   = 6.625e-27
C   = 3e10
K   = 1.38e-16
amu = 1.6605402e-24 #atomic mass unit
yr  = 365.*24.*60.*60.

# -------------------------------------------------------------------------
# Experiment setup
# -------------------------------------------------------------------------
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-f", "--file",
                  dest="file",
                  help="file with data")
(options, args) = parser.parse_args()

inputs = options.file                  
#filename = 'inputs_test.dustpol'
q = genfromtxt(inputs,skip_header=0,dtype=None,names=['names','params'],\
    comments='!',usecols=(0,1),encoding='utf=8')
params     = q['params']
cloud      = params[0]
ratd       = params[1]
working_lam= eval(params[2])
try:
    U      = eval(params[3])
    if isinstance(U,list) or isinstance(U,tuple):
        U  = array(U)
    else:
        U  = array([U])
except:
    U      = params[3]
if isinstance(U,int) or isinstance(U,float):
    U      = array([U])
gamma      = eval(params[4])
lambA      = eval(params[5])*1e-4 #cm
dpc        = params[6]+'kpc'
n_gas      = eval(params[7])
T_gas      = eval(params[8])
Avrange    = eval(params[9])
mgas       = 1.3*amu #90%H + 10%He
if isinstance(Avrange,int) or isinstance(Avrange,float):
    Avrange = array([Avrange])
dust_type  = params[10]
amin       = eval(params[11])*1e-4 #cm
amax       = eval(params[12])*1e-4 #cm
f_max      = eval(params[13])
Tdust      = eval(params[14])
if isinstance(Tdust,int) or isinstance(Tdust,float):
    Tdust = array([Tdust])
rho        = eval(params[15])
alpha      = eval(params[16])
Smax       = eval(params[17])
dust_to_gas_ratio=eval(params[18])
GSD_law    = params[19]
power_index= eval(params[20])
parallel   = eval(params[21])
n_jobs     = eval(params[22])
overwrite  = params[23]
if overwrite=='No' or overwrite=='no':
    checking=True
else:
    checking=False

# -------------------------------------------------------------------------
# output direction
# -------------------------------------------------------------------------
if ratd=='on':
    if GSD_law=='MRN':
        dir_out = 'output/disruption'+str(ratd)+'/fmax='+str(around(f_max,2))+'/MRN_'+str(power_index)+'/Smax='+'{:.0e}'.format(Smax)+'/'#1e7/'#+str(Smax)+'/'
    elif GSD_law=='WD01' or GSD_law=='DL07':
        dir_out = 'output/disruption'+str(ratd)+'/fmax='+str(around(f_max,2))+'/'+str(GSD_law)+'/Smax='+'{:.0e}'.format(Smax)+'/'#1e7/'#+str(Smax)+'/'
    else:
        log.error("Grain-size distribution is not found! [\033[1;5;7;91m failed \033[0m]")
        raise IOError('Grain size distribution')
    ##endif
elif ratd=='off':
    if GSD_law=='MRN':
        dir_out = 'output/disruption'+str(ratd)+'/fmax='+str(around(f_max,2))+'/MRN_'+str(power_index)+'/'
    elif GSD_law=='WD01':
        dir_out = 'output/disruption'+str(ratd)+'/fmax='+str(around(f_max,2))+'/WD01/'
    else:
        log.error("Grain-size distribution is not found! [\033[1;5;7;91m failed \033[0m]")
        raise IOError('Grain size distribution')
else:
    log.error("Please check RATD on/off! [\033[1;5;7;91m failed \033[0m]")
    raise IOError('RATD on/off')
##endif
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# -------------------------------------------------------------------------
# radiation
# -------------------------------------------------------------------------
# ISRF radiation strength
#if cloud == 'MC':
u_ISRF = 8.64e-13 # typical interstellar radiation field
#else:
#    import rad_func
#    if dpc!='5kpc':
#        log.error('In this version, dpc should be 5kpc! -- SORRY')
#        raise IOError
#    u_ISRF = rad_func.uISRF(dpc) # interstellar radiation field at different distance

# Radiation Field from Mathis 83'
if cloud=='Mathis83':
    import rad_func
    Urange = zeros((len(Avrange)))
    lrange = zeros((len(Avrange)))
    for iv in range(len(Avrange)):
        Urange[iv] = np.round(rad_func.LambU(Avrange[iv],u_ISRF,dpc)[0],2)
        lrange[iv] = np.round(rad_func.LambU(Avrange[iv],u_ISRF,dpc)[1],5)
    Av    = Avrange[iu]
    print('[Av='+str(Av)+']')
# Radiation Field from inputs or thermal equilibrium
if isinstance(U,ndarray):
    Urange = U
else: 
    if (U=='Tdust'):
        #Urange = lambda a: (array(Tdust)/16.4)**(6) * (a/1e-5)**(6/15) ##correct formula
        #for small a
        if Tdust==0:
            log.error('Please give values of Tdust [\033[1;5;7;91m failed \033[0m]')
            raise IOError('Values of Tdust')
        else:
            Urange = (array(Tdust)/16.4)**(6)
    else:
        log.error('Values of U are not regconized! [\033[1;5;7;91m failed \033[0m]')
        raise IOError('Values of U')

# -------------------------------------------------------------------------
# DDSCAT
# -------------------------------------------------------------------------
##[!Warning] we don't incorporate the DDSCAT code, but we instead tabulate for a
#set of radiation strength (U)
DDSCAT_path = 'data/'
all_folders = os.listdir(DDSCAT_path)
DDSCAT_tab  = [all_folders[i] for i in range(len(all_folders)) if 'U=' in all_folders[i]]
Urange_DDSCAT=[eval(re.findall("\d+\.\d+", DDSCAT_tab[i])[0]) for i in range(len(DDSCAT_tab))]
Urange_DDSCAT.sort()
##Checking for the input Urange
for i in range(len(Urange)):
    if Urange[i]<min(Urange_DDSCAT) or Urange[i]>max(Urange_DDSCAT):
        log.error("Your value of U=%.3f is beyond the DDSCAT's tabulation [%.1f,%.1f] [\033[1;5;7;91m failed \033[0m]"\
                    %(Urange[i],min(Urange_DDSCAT),max(Urange_DDSCAT)))
        raise IOError('Value of U')

# -------------------------------------------------------------------------
# INFORMATION
# -------------------------------------------------------------------------
##ANSI color
class bcolors:
    red    = '\033[31m'
    green  = '\033[32m'
    yellow = '\033[33m'
    blue   = '\033[34m'
    purple = '\033[35m'
    cyan   = '\033[36m'
    HEADER = '\033[1;95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print('!-----------------------------------------------------!                                     ')
print('!'+f"{bcolors.HEADER}               WELCOME to DustPOL (v1.3)        {bcolors.ENDC}"+'     !')
print('!'+f"{bcolors.blue}              by VARNET Theoretical team         {bcolors.ENDC}"+'    !')
print('!-----------------------------------------------------!')
print('\n')
print(f"{bcolors.UNDERLINE}{bcolors.green}                      YOUR INPUTs                       {bcolors.ENDC}")

log.info('Cloud     : %s'%(cloud))
log.info('RATD      : %s'%(ratd))
log.info('U         : %s'%(Urange))
log.info('gamma     : %.1f'%(gamma))
log.info('ngas      : %.3f [cm-3]'%(n_gas))
log.info('dust-type : %s'%(dust_type))
log.info('amax      : %.3f [um]'%(amax*1e4))
log.info('fmax      : %.3f'%(f_max))
log.info('GSD       : %s and with power-index: %.1f'%(GSD_law,power_index))
print('\n')
print(f"{bcolors.UNDERLINE}{bcolors.green}                        DustPOL                         {bcolors.ENDC}")