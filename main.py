from functions import *
#from FDLF_2again import *
#from DCPF import *
'''
Main file
Newton Rhapson:
    - builds matrices and iterates until convergence
        
'''

def main():
    filepath = '/Users/gracedepietro/Desktop/4205/project/PowerFlow/'
    #change these for the NR methods for which type of qlmit method, which bus/es its on, and the limit value
    conv_crit = 0.0000001
    qlim_type = 'each' #each #none #change this to none to do regular NR wihtout qlims
    filenameNR = filepath + 'ex_nr_ex1.xlsx'
    #newtonRhapson(conv_crit, qlim_type, filenameNR)
    filenameFDLF = filepath + 'ex_nr_ex1.xlsx'
    qlimType = 'each' #'none' #'each' # each or none for including or not including q limits
    it_type = 'end_it' #'mid_it' #'end_it' # end or mid iteration method
    fastDLF(conv_crit, 'each', filenameFDLF, 'end_it')
    #fastDLF(conv_crit, 'none', filenameFDLF, it_type)
    filenameDCPF = filepath + 'ex_nr_ex2.xlsx'
    #printDCPF(filenameDCPF)
    filenameDLF = filepath + 'ex_nr_ex2.xlsx'
    # decoupledLoadFlow(conv_crit, filenameDLF)






if __name__ == "__main__":
    main()