from functions import *

'''
Main file
Newton Rhapson:
    - builds matrices and iterates until convergence
        
'''

def main():
    conv_crit = 0.000001
    newtonRhapson(conv_crit)
    FastDecoupled(conv_crit)






if __name__ == "__main__":
    main()