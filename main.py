from functions import *
#from DCPF import *
'''
Main file
Newton Rhapson:
    - builds matrices and iterates until convergence
        
'''

def main():
    conv_crit = 0.000001
    qlim_type = 'none' #each #end
    qbus = 0 #1 #[1 2]
    num_lims = 0 #1 #2
    qlim_val = 0 #0.5
    newtonRhapson(conv_crit, qlim_type, qbus, num_lims, qlim_val)
    FastDecoupled(conv_crit)
    FastDecoupled(conv_crit)
    printDCPF()
    decoupledLoadFlow(conv_crit)






if __name__ == "__main__":
    main()