from functions import *

'''
Main File for TET 4205 Power Flow final course project
Fall 2023
Study Group 2
'''

def main():
    # Change Filepath for use on personal computer
    filepath = ''
    # Change the convergence criteria
    conv_crit = 0.0000001

    ############################ Newton-Raphson ###########################################
    # Change the qlim_type to be "on" or "none" to run with reactive limits or not
    qlim_type = 'none' #on #none #change this to none to do regular NR wihtout qlims
    # Change the filenameNR to whichever excel
    filenameNR = filepath + 'test_system_2.xlsx'
    # Comment or uncomment to run
    #newtonRhapson(conv_crit, qlim_type, filenameNR)

    ############################ Decoupled load flow ###########################################
    # Change the filenameDLF to whichever excel
    filenameDLF = filepath + 'test_system_2.xlsx'
    # Comment or uncomment to run
    #decoupledLoadFlow(conv_crit, filenameDLF)

    ############################ Fast Decoupled load flow ###########################################
    # Change the filenameFDLF to whichever excel
    filenameFDLF = filepath + 'test_system_2.xlsx'
    # Change qlim_type to be 'on' or 'none' to run with or without reactive limits
    qlimTypeFast = 'none' #'none' #'on' # each or none for including or not including q limits
    # Change it_type to be 'end_it' or 'mid_it' to run with the change at the end of the iteration or mid iteration
    it_type = 'end_it' #'mid_it' #'end_it' # end or mid iteration method
    # Comment or uncomment to run
    #fastDLF(conv_crit, qlimTypeFast, filenameFDLF, it_type)

    ############################ DC Power FLow ###########################################
    # Change the filenameDCPF to whichever excel
    filenameDCPF = filepath + 'test_system_2.xlsx'
    # Comment or uncomment to run
    printDCPF(filenameDCPF)



if __name__ == "__main__":
    main()