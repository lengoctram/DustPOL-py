import numpy as np
import os
import pkg_resources
from . import constants 
from .constants import au,amu
from astropy import log
d = pkg_resources.get_distribution('DustPOL_py')

#path= '/Users/lengoctram/Documents/postdoc/MPIfR/research/dustpol/src_v1.5/DustPOL-RAT+MRAT+2-layer_original/'#os.getcwd()
## -------------------------------------------------------------------------
## Experiment setup
##-------------------------------------------------------------------------
# from optparse import OptionParser
# parser = OptionParser()
# parser.add_option("-f", "--file",
#                   dest="file",
#                   help="file with data")
# (options, args) = parser.parse_args()

# inputs = options.file                 

class input():
    def __init__(self,input_file):
        self.input_file = input_file
        self.path = d.location+'/DustPOL_py/'#sys.path.insert(0, abspath(join(dirname(__file__), '.')))#os.getcwd()+'/'
        inputs = self.input_file#self.path+self.input_file
        q = np.genfromtxt(inputs,skip_header=1,dtype=None,names=['names','params'],\
        comments='!',usecols=(0,1),encoding='utf=8')
        params     = q['params']
        self.output_dir = params[0]
        self.ratd       = eval(params[1])
        self.p          = eval(params[2])
        self.rin        = eval(params[3]) * au #cm
        self.rout       = eval(params[4]) * au #cm
        self.rflat      = eval(params[5]) * au #cm
        self.nsample    = eval(params[6])
        self.U          = eval(params[7])
        self.gamma      = eval(params[8])
        self.mean_lam   = eval(params[9])*1e-4 #cm
        self.dpc        = params[10]+'kpc'
        self.ngas       = eval(params[11])
        self.Tgas       = eval(params[12])
        self.Avrange    = eval(params[13])
        self.mgas       = 1.3*amu #90%H + 10%He
        self.dust_type  = params[14]
        self.amin       = eval(params[15])*1e-4 #cm
        self.amax       = eval(params[16])*1e-4 #cm
        self.Tdust      = eval(params[17])
        self.rho        = eval(params[18])
        self.alpha      = eval(params[19])
        self.Smax       = eval(params[20])
        self.dust_to_gas_ratio=eval(params[21])
        self.GSD_law    = params[22]
        self.power_index= eval(params[23])
        self.RATalign = params[24].lower() # RAT or MRAT
        self.f_max    = eval(params[25])
        self.B_angle  = eval(params[27])*np.pi/180 # rad.
        if self.RATalign=='mrat':
            self.Bfield = eval(params[26])
            self.Ncl    = eval(params[28])
            self.phi_sp = eval(params[29])
            self.fp     = eval(params[30])
        else:
            self.Bfield = np.nan
            self.Ncl    = np.nan
            self.phi_sp = np.nan
            self.fp     = np.nan
        # model_layer = eval(params[31])

        self.parallel   = eval(params[35])
        if (self.parallel):
            self.cpu     = eval(params[36])
            if (self.cpu==-1): 
                self.max_workers=os.cpu_count()#None
            else:
                if not isinstance(self.cpu,int): 
                    raise IOError('cpu number must be an integer!')
                if (self.cpu>os.cpu_count()):
                    log.warning('the input value of cpu > your cpu --> use all your cpu cores')
                    self.max_workers=os.cpu_count()
                else:
                    self.max_workers=self.cpu

        # overwrite  = params[37]
        # if overwrite=='No' or overwrite=='no':
        #     checking=True
        # else:
        #     checking=False
        self.u_ISRF = 8.64e-13 #(ergcm-3) typical interstellar radiation field

class output():
    def __init__(self,parent,filename,data):
        self.U=parent.U
        self.alpha=parent.alpha
        self.path =parent.path
        # subpath = path+'output/starless/astrodust/'#U=%.2f'%(self.U)+'/'
        # if not os.path.exists(subpath):
        #     os.mkdir(subpath)
        # subsubpath = subpath+'U=%.2f_alpha=%.4f'%(self.U,self.alpha)+'/Av_fixed_amax/'
        # if not os.path.exists(subsubpath):
        #     os.mkdir(subsubpath)

        # subpath = 'output/'#self.path+'output/'
        subpath =parent.output_dir+'/'
        if not os.path.exists(subpath):
            os.makedirs(subpath)
        self.filename=subpath+filename
        self.ngas=parent.ngas
        self.mean_lam=parent.mean_lam
        self.gamma=parent.gamma
        self.amax=parent.amax
        self.dust_type=parent.dust_type
        self.data=data
        try:
            self.Av_array=parent.Av_array
        except:
            self.Av_array=None

        # output_abs = path+'amax=%.2f'%(self.amax*1e4)+'_abs.dat'
        # output_emi = path+'amax=%.2f'%(self.amax*1e4)+'_emi.dat'

        self.file_save()

    def file_save(self):
        f=open(self.filename,'w')
        f.write('U=%.3f \n'%self.U)
        f.write('ngas=%.3e (cm-3) \n'%self.ngas)
        f.write('mean_lam=%.3f (um) \n'%(self.mean_lam*1e4))
        f.write('gamma=%.3f \n'%self.gamma)
        f.write('amax=%.3f (um) \n'%(self.amax*1e4))
        f.write('dust_composition=%s \n'%self.dust_type)
        f.write('! \n')
        if self.Av_array is None:
            f.write('Av= ')
            f.write(",".join(str("{:.3f}".format(iAv)) for iAv in self.Av_array) + "\n")
            f.write('! \n')
        else:
            if isinstance(self.Av_array,float):
                f.write('Av= %.3f'%self.Av_array + "\n")
                f.write('! \n')
            elif isinstance(self.Av_array,np.ndarray):
                f.write('Av= ')
                f.write(",".join(str("{:.3f}".format(iAv)) for iAv in self.Av_array) + "\n")
                f.write('! \n')
        #keys=sorted(data_save.keys())
        keys=list(self.data.keys())
        print(' \t\t'.join(keys), end="\n",file=f)
        for i in range(len(self.data[keys[0]])):
            line=''
            for k in keys:
                # line=line+str(self.eformat(self.data[k][i],4,2))+'\t '
                line=line+str("{:.3e}".format(self.data[k][i]))+'\t '

            print(line,end="\n",file=f)
        f.close()

    def eformat(self,f, prec, exp_digits):
        s = "%.*e"%(prec, f)
        mantissa, exp = s.split('e')
        # add 1 to digits as 1 is taken by sign +/-
        return "%se%+0*d"%(mantissa, exp_digits+1, int(exp))
