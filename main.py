from functions import *
#from FDLF_2again import *
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
    filenameNR = 'C:/Users/Uxue/Desktop/TETE4205/PowerFlow/ex_nr_ex2.xlsx'
    #newtonRhapson(conv_crit, qlim_type, filenameNR)
    filenameFDLF = 'C:/Users/Uxue/Desktop/TETE4205/PowerFlow/ex_nr_ex2.xlsx'
    fastDLF(conv_crit, qlim_type, filenameFDLF)
    filenameDCPF = 'C:/Users/Uxue/Desktop/TETE4205/PowerFlow/ex_nr_ex2.xlsx'
    #printDCPF(filenameDCPF)
    #filenameDLF = 'C:/Users/Uxue/Desktop/TETE4205/PowerFlow/ex_nr_ex2.xlsx'
    # decoupledLoadFlow(conv_crit, filenameDLF)






if __name__ == "__main__":
    main()