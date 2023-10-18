import numpy as np
import matplotlib.pyplot as plt
from classes import *
from functions import *
import pandas as pd

'''
Variables: 
    numbers:
        busnum
        knownNum
    matrices:
        knowns
        xmat
        
'''

def main():
    #read excel file
    initial = pd.read_excel('/Users/gracedepietro/Desktop/4205/project/PowerFlow/ex_nr.xlsx', sheet_name='initial', index_col='bus_num')
    #extract number of buses
    busnum = len(initial.index)
    #extract v and theta values and sub in 0 or 1 for first guess
    v_list = initial.loc[:, 'V'].to_numpy()
    v_list[np.isnan(v_list)] = 1
    v_list = np.array(v_list)
    v_list = v_list.astype('float64')
    t_list = initial.loc[:, 'T'].to_numpy()
    t_list[np.isnan(t_list)] = 0
    t_list = np.array(t_list)
    t_list = t_list.astype('float64')
    #extract p and q list
    #adds NANs to spots where there is no initial
    p_list = initial.loc[:, 'P'].to_numpy()
    q_list = initial.loc[:, 'Q'].to_numpy()
    print("Initial things we know: ")
    print("Number of buses: ", busnum)
    print("List of Vs: ", v_list)
    print("List of Ts: ", t_list)
    print("List of Ps: ", p_list)
    print("List of Qs: ", q_list)

    numP = initial.loc[:, 'P'].count()
    numQ = initial.loc[:, 'Q'].count()
    numT = numP
    numV = numQ
    knownnum = numP + numQ
    line_z = pd.read_excel('/Users/gracedepietro/Desktop/4205/project/PowerFlow/ex_nr.xlsx', sheet_name='line_imp')

    print("Number of Ps: ", numP)
    print("Number of Qs: ", numQ)
    print("Number of knowns: ", knownnum)

    knowns = [VarMat() for i in range(int(knownnum))]
    xmat = [VarMat() for j in range(int(knownnum))]

    printMat(knownnum, xmat)
    getInitMats(xmat, knowns, p_list, q_list, busnum)
    setInitGuess(knownnum, xmat)
    print("initial empty xmat")
    printMat(knownnum, xmat)
    #printMat(knownNum, knowns)

    #xmat = setInitGuess(knownNum, xmat)

    #printMat(busnum, xmat)

    yBus = [[VarMat() for i in range(int(busnum))] for j in range(int(busnum))]
    zBus = [[complex(0, 0) for i in range(int(busnum))] for j in range(int(busnum))]
    getZYbus(busnum, yBus, zBus, line_z)

    print(zBus)

    calcYbus(busnum, yBus, zBus)
    printMultiMat(busnum, yBus, False)
    print("first element of y bus real part is: ", yBus[0][0].val.real)
    jacobian = [[JacElem() for i in range(int(knownnum))] for j in range(int(knownnum))]
    nameJacElem(knownnum, knowns, xmat, jacobian)
    print("empty jacobian: ")
    printMultiMat(knownnum, jacobian, True)
    #iterate #1
    iterate(knownnum, jacobian, yBus, t_list, v_list, knowns, xmat, busnum, numT, numV)
    print("filled jacobian: ")
    printMultiMat(knownnum, jacobian, True)
    print("New voltage angles and magnitudes")
    printMat(knownnum, xmat)
    #iterate #2
    iterate(knownnum, jacobian, yBus, t_list, v_list, knowns, xmat, busnum, numT, numV)
    print("filled jacobian: ")
    printMultiMat(knownnum, jacobian, True)
    print("New voltage angles and magnitudes")
    printMat(knownnum, xmat)






if __name__ == "__main__":
    main()