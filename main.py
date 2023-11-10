from functions import *
from FDLF_jac_try import *
#from DCPF import *
'''
Main file
Newton Rhapson:
    - builds matrices and iterates until convergence
        
'''

def main():
    #change these for the NR methods for which type of qlmit method, which bus/es its on, and the limit value
    conv_crit = 0.0000001
    qlim_type = 'none' #none #end #change this to none to do regular NR wihtout qlims
    filenameNR = '/Users/gracedepietro/Desktop/4205/project/PowerFlow/ex_nr_ex1.xlsx'
    newtonRhapson(conv_crit, qlim_type, filenameNR)
    filenameFDLF = '/Users/gracedepietro/Desktop/4205/project/PowerFlow/ex_nr_ex1.xlsx'
    # FastDecoupled(conv_crit, filenameFDLF)
    # FastDecoupled(conv_crit)
    filenameDCPF = '/Users/gracedepietro/Desktop/4205/project/PowerFlow/ex_nr_ex1.xlsx'
    # printDCPF(filenameDCPF)
    filenameDLF = '/Users/gracedepietro/Desktop/4205/project/PowerFlow/ex_nr_ex1.xlsx'
    # decoupledLoadFlow(conv_crit, filenammeDLF)






if __name__ == "__main__":
    main()