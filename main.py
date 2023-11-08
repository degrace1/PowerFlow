from functions import *
#from DCPF import *
'''
Main file
Newton Rhapson:
    - builds matrices and iterates until convergence
        
'''

def main():
    #change these for the NR methods for which type of qlmit method, which bus/es its on, and the limit value
    conv_crit = 0.0000001
    qlim_type = 'each' #none #end #change this to none to do regular NR wihtout qlims
    newtonRhapson(conv_crit, qlim_type)
    #newtonRhapson(conv_crit, 'none', [], 0)
    # FastDecoupled(conv_crit)
    # FastDecoupled(conv_crit)
    #printDCPF()
    # decoupledLoadFlow(conv_crit)
    #newtonRhapson(conv_crit, 'none')





if __name__ == "__main__":
    main()