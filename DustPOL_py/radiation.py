import numpy as np
import os,re
from astropy import log
# -------------------------------------------------------------------------
# Temperature Distribution (dPdT) Pre-calculated by DustEM code
# -------------------------------------------------------------------------
##[!Warning] we don't calculate here dPdT, but we instead tabulate dPdT for a
#set of radiation strength (U)
class radiation_retrieve():
    def __init__(self,parent):
        self.U = parent.U
        self.path=parent.path
    def retrieve(self):
        tempdist_path = self.path+'data/'
        all_folders = os.listdir(tempdist_path)
        tempdist_tab  = [all_folders[i] for i in range(len(all_folders)) if 'U=' in all_folders[i]]
        self.Urange_tempdist=[eval(re.findall("\d+\.\d+", tempdist_tab[i])[0]) for i in range(len(tempdist_tab))]
        self.Urange_tempdist.sort()

        ##MAKE A TRICK
        if self.U < min(self.Urange_tempdist):
            log.warning('*** Your value of U=%.3f < Umin=%.3f --> \033[1;5;33m set U == %.3f \033[0m'%(self.U,min(self.Urange_tempdist),min(self.Urange_tempdist)))
            self.U=min(self.Urange_tempdist)
        elif self.U> max(self.Urange_tempdist):
            log.warning('*** Your value of U=%.3f > Umax=%.3f --> \033[1;5;33m set U == %.3f \033[0m'%(self.U,max(self.Urange_tempdist),max(self.Urange_tempdist)))
            self.U=max(self.Urange_tempdist)
        else:
            self.U=self.U

        idx = abs(np.array(self.Urange_tempdist)-self.U).argmin()
        ##update radiation field
        self.U = self.Urange_tempdist[idx]
        return self.U
        #