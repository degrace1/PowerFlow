from functions import *
from FDLF_jac_try import *
#from DCPF import *
'''
Main file
Newton Rhapson:
    - builds matrices and iterates until convergence
        
'''

def main():
    conv_crit = 0.000001
    newtonRhapson(conv_crit, False, '', 0, 0, 0)
    #FastDecoupled(conv_crit)
    newtonRhapson(conv_crit)
    FastDecoupled(conv_crit)
    printDCPF()






if __name__ == "__main__":
    main()