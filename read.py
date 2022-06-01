import numpy as np
from numpy import *

# read text data
def readD(filename,skipH,N_par):
    file=open(filename,'r')
    data=[]
    for ln in file:
        data.append(ln.split())
    dataf = np.zeros([N_par,len(data)-skipH])
    for j in range(skipH,len(data)):
        for i in range(N_par):
            dataf[i,j-skipH] = data[j][i]
    return dataf

# read text data - multile data combined
def readDC(filename,Header,skip,Nd,Nline,N_par):
    file=open(filename,'r')
    data=[]
    for ln in file:
        data.append(ln.split())
    data = data[Header:len(data)]
    dataf = np.zeros([N_par,Nline,Nd])
    for i in range(Nd):
        for j in range(Nline):
            for k in range(N_par):
                dataf[k,j,i] = data[i*(Nline+skip)+skip+j][k]
    return dataf

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()