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
    #newtonRhapson(conv_crit,1,1,10)
    FastDecoupled(conv_crit)
    #FDLF_jac(conv_crit,1,1,10)






if __name__ == "__main__":
    main()