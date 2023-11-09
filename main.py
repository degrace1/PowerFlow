from functions import *
from FDLF_jac_try import *
#from DCPF import *
'''
Main file
Newton Rhapson:
    - builds matrices and iterates until convergence
        
'''

def main():
    #ex_nr_ex2.xlsx is the test system 2 for our group
    #change these for the NR methods for which type of qlmit method, which bus/es its on, and the limit value
    conv_crit = 0.000001
    qlim_type = 'none' #each #change this to none to do regular NR wihtout qlims
    filenameNR = '/Users/gracedepietro/Desktop/4205/project/PowerFlow/ex_nr_ex1.xlsx'
    newtonRhapson(conv_crit, qlim_type, filenameNR)
    filenameFDLF = '/Users/gracedepietro/Desktop/4205/project/PowerFlow/ex_nr_ex1.xlsx'
    # FastDecoupled(conv_crit, filenameFDLF)
    filenameDCPF = '/Users/gracedepietro/Desktop/4205/project/PowerFlow/ex_nr_ex2.xlsx'
    # printDCPF(filenameDCPF)
    filenameDLF = '/Users/gracedepietro/Desktop/4205/project/PowerFlow/ex_nr_ex2.xlsx'
    # decoupledLoadFlow(conv_crit, filenameDLF)






if __name__ == "__main__":
    main()