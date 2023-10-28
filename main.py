from functions import *
#from DCPF import *
'''
Main file
Newton Rhapson:
    - builds matrices and iterates until convergence
        
'''

def main():
    #change these for the NR methods for which type of qlmit method, which bus/es its on, and the limit value
    conv_crit = 0.000001
    qlim_type = 'each' #none #end #change this to none to do regular NR wihtout qlims
    qbus = [3] #[1 2]
    qlim_val = 0.4 #0
    newtonRhapson(conv_crit, qlim_type, qbus, qlim_val)
    # FastDecoupled(conv_crit)
    # FastDecoupled(conv_crit)
    # printDCPF()
    # decoupledLoadFlow(conv_crit)






if __name__ == "__main__":
    main()