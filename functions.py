# Functions file for Power FLow 4205 Project
from classes import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmath

"""
Function: get initial matrices
This function asks the user to input the initial knowns (P, Q) and their associated value.
If the user inputs a P, a T for delta/angle will be added to the unknown matrix. If a Q is
entered, a V will be added.
Parameters:
    busnum - total number of buses
    xmat - empty matrix of size busnum rows and 1 column (unknowns matrix)
    knowns - empty matrix of size busnum rows and 1 column (knowns matrix)
    p_list - p vals
    q_list - q vals
Returns:
    known - the filled known matrix with name and values
    xmat - the filled unknown matrix with name only

"""
def getInitMats(xmat, knowns, p_list, q_list, busnum):
    sumcount = 0
    for i in range(busnum):
        # check if there is a value for initial P
        # add a T to the xmat and a P to the knowns
        if np.isnan(p_list[i]) == False:
            xmat[sumcount].name = "T" + str(i + 1)
            knowns[sumcount].name = "P" + str(i + 1)
            knowns[sumcount].val = p_list[i]
            sumcount += 1
            print(sumcount)
    newcount = 0
    for j in range(busnum):
        # check if there is a value for initial Q
        # add a V to the xmatrix and a Q to the knowns
        if np.isnan(q_list[j]) == False:
            xmat[newcount + sumcount].name = "V" + str(j + 1)
            knowns[newcount + sumcount].name = "Q" + str(j + 1)
            knowns[newcount + sumcount].val = q_list[j]
            newcount += 1


""" NOT USED - OBSOLETE
Function: Set initial guess
This function sets the initial guesses of 1.0pu for voltage and 0 for angle
Parameters:
    knownnum - number of knowns in the system
    xmat - matrix of unknowns with only names
Returns
    xmat - matrix of unknowns with names and initial guesses
"""
def setInitGuess(knownNum, xmat):
    for i in range(int(knownNum)):
        if "V" in xmat[i].name:
            xmat[i].val = 1
        elif "T" in xmat[i].name:
            xmat[i].val = 0
    return xmat


"""
Function: print matrix
This will print a known or unknown matrix (rows amount = busnum, column = 1)
Parameters:
    num - number of elem for rows in matrix
    xmat - matrix
Returns:
    nothing, just print
"""
def printMat(num, xmat):
    for i in range(int(num)):
        print(xmat[i].name + ", " + str(xmat[i].val))


"""
Function: print matrix
This will print a matrix (probably Ybus) (rows & columns amount = busnum)
Parameters:
    num - number of elem for rows and columns in matrix
    xmat - matrix
Returns:
    nothing, just print
"""
def printMultiMat(num, mat, jac):
    for i in range(int(num)):
        for j in range(int(num)):
            if jac == False:
                print(mat[i][j].name + ", " + str(mat[i][j].val))
            else:
                print(mat[i][j].name + ", " + str(mat[i][j].val), ", ", str(mat[i][j].type))


"""
Function: get zy bus
This will ask the user to input the known values of the line impedances and create a matrix of them.
This will also set the names of the Ybus matrix things (just since I already put the for loops).
Asks for a and b parts of the complex number.
Parameters:
    busnum - number of nuses for matrix size indeces
    yBus - matrix with yBus names and vals
    zbus - empty matrix of size busnumxbusnum
Returns:
    yBus - filled with names of the ybus for later use
    zBus - filled with values from user (complex #s)
"""
def getZYbus(busnum, yBus, zBus, z_imp):
    countz = 0
    for i in range(int(busnum)):
        for j in range(int(busnum)):
            yBus[i][j].name = "Y" + str(i + 1) + str(j + 1)
            # ask for "zbus" values
            if j != i:
                if zBus[j][i] != complex(0, 0) or zBus[j][i] != 0:
                    zBus[i][j] = zBus[j][i]
                else:
                    if z_imp.loc[countz, 'line'] == int(str(i + 1) + str(j + 1)):
                        a = z_imp.loc[countz, 'R']
                        b = z_imp.loc[countz, 'X']
                        countz += 1
                        zBus[i][j] = complex(a, b)


"""
Function: calculate yBus
This will take the line impedances and find the yBus
Parameters:
    busnum - number of buses for matrix size
    yBus - matrix of size busnumxbusnum of named yBus but no values yet
    zBus - line impedances
Returns:
    yBus - filled in with values
"""
def calcYbus(busnum, yBus, zBus):
    num = int(busnum)
    for i in range(num):  # row number
        for j in range(num):  # column number
            # calc diagnol
            if i == j:
                sum = 0
                for k in range(num):
                    # loop through other numbers and check if its not the same as self
                    # check if impedance i,k isn't zero (avoid inf zero)
                    if k != i and zBus[i][k] != complex(0, 0) and zBus[i][k] != 0:
                        # if true add that value to sum
                        sum += 1 / zBus[i][k]
                if sum == 0:  # set to zero
                    yBus[i][j].val = complex(0, 0)
                else:  # set to value
                    yBus[i][j].val = sum
            else:  # if its not a diagnol element
                # check that its not zero to avoid inf zero
                if zBus[i][j] != complex(0, 0) and zBus[i][j] != 0:
                    yBus[i][j].val = -1 / zBus[i][j]
                else:
                    yBus[i][j].val = complex(0, 0)


'''
Function: calculate uij
'''
def uij(gij, bij, thetai, thetaj):
    return (gij * np.sin(thetai - thetaj) - bij * np.cos(thetai - thetaj))


'''
Function: calculate tij
'''
def tij(gij, bij, thetai, thetaj):
    return (gij * np.cos(thetai - thetaj) + bij * np.sin(thetai - thetaj))


'''
Function: calculate P value
'''
#FIXME: added knowns matrix so we can just change the value directly inside? or not because we need to add to old p
def calcPVal(num, yBus, knownNum, T, V):
    num = int(num)
    p = V[num] ** 2 * yBus[num][num].val.real
    for j in range(int(knownNum)):
        p -= V[num] * V[j] * tij(yBus[num][j].val.real, yBus[num][j].val.imag, T[num], T[j])
    return p


'''
Function: calculate Q value
'''
def calcQVal(num, yBus, knownNum, T, V):
    num = int(num)
    q = -V[num] ** 2 * yBus[num][num].val.imag
    for j in range(knownNum):
        q -= V[num] * V[j] * uij(yBus[num][j].val.real, yBus[num][j].val.imag, T[num], T[j])
    return q


'''
Function: calculate partial derivative dPi / dTi
'''
def dpidti(i, V, yBus, T, knownNum):
    i = int(i)
    sum = 0
    for j in range(knownNum):
        if j != i:
            sum += V[j] * uij(yBus[i][j].val.real, yBus[i][j].val.imag, T[i], T[j])
    return sum * V[i]


'''
Function: calculate partial derivative dPi / dTj
'''
def dpidtj(i, j, V, yBus, T):
    i = int(i)
    j = int(j)
    return -V[i] * V[j] * uij(yBus[i][j].val.real, yBus[i][j].val.imag, T[i], T[j])


'''
Function: calculate partial derivative dPi / dVi
'''
def dpidvi(i, V, yBus, T, knownnum):
    i = int(i)
    sum = 2 * V[i] * yBus[i][i].val.real
    for j in range(knownnum):
        if j != i:
            sum += V[j] * tij(yBus[i][j].val.real, yBus[i][j].val.imag, T[i], T[j])
    return sum


'''
Function: calculate partial derivative dPi / dQj
'''
def dpidvj(i, j, V, yBus, T):
    i = int(i)
    j = int(j)
    return -V[i] * tij(yBus[i][j].val.real, yBus[i][j].val.imag, T[i], T[j])


'''
Function: calculate partial derivative dQi / dTi

'''
def dqidti(i, V, yBus, T, knownNum):
    i = int(i)
    sum = 0
    for j in range(knownNum):
        if j != i:
            sum += V[j] * tij(yBus[i][j].val.real, yBus[i][j].val.imag, T[i], T[j])
    return sum * -V[i]


'''
Function: calculate partial derivative dQi / dTj

'''
def dqidtj(i, j, V, yBus, T):
    i = int(i)
    j = int(j)
    return V[i] * V[j] * tij(yBus[i][j].val.real, yBus[i][j].val.imag, T[i], T[j])


'''
Function: calculate partial derivative dQi / dVi
'''
def dqidvi(i, V, yBus, T, knownnum):
    i = int(i)
    sum = -2 * V[i] * yBus[i][i].val.imag
    for j in range(knownnum):
        if j != i:
            sum += -V[j] * uij(yBus[i][j].val.real, yBus[i][j].val.imag, T[i], T[j])
    return sum


'''
Function: calculate partial derivative dQi / dVj
'''
def dqidvj(i, j, V, yBus, T):
    i = int(i)
    j = int(j)
    x = (-V[i] * uij(yBus[i][j].val.real, yBus[i][j].val.imag, T[i], T[j]))
    return x


'''
Function: name jacobian element
'''
def nameJacElem(knownnum, knowns, xmat, jacobian):
    for i in range(knownnum):
        for j in range(knownnum):
            # Ps
            if knowns[i].name[0] == 'P' and xmat[j].name[0] == 'T':
                jacobian[i][j].name = 'dp' + str(knowns[i].name[1]) + 'dt' + str(xmat[j].name[1])
                if knowns[i].name[1] == xmat[j].name[1]:
                    jacobian[i][j].type = 'dpidti'
                else:
                    jacobian[i][j].type = 'dpidtj'
            if knowns[i].name[0] == 'P' and xmat[j].name[0] == 'V':
                jacobian[i][j].name = 'dp' + str(knowns[i].name[1]) + 'dv' + str(xmat[j].name[1])
                if knowns[i].name[1] == xmat[j].name[1]:
                    jacobian[i][j].type = 'dpidvi'
                else:
                    jacobian[i][j].type = 'dpidvj'
            # Qs
            if knowns[i].name[0] == 'Q' and xmat[j].name[0] == 'T':
                jacobian[i][j].name = 'dq' + str(knowns[i].name[1]) + 'dt' + str(xmat[j].name[1])
                if knowns[i].name[1] == xmat[j].name[1]:
                    jacobian[i][j].type = 'dqidti'
                else:
                    jacobian[i][j].type = 'dqidtj'
            if knowns[i].name[0] == 'Q' and xmat[j].name[0] == 'V':
                jacobian[i][j].name = 'dq' + str(knowns[i].name[1]) + 'dv' + str(xmat[j].name[1])
                if knowns[i].name[1] == xmat[j].name[1]:
                    jacobian[i][j].type = 'dqidvi'
                else:
                    jacobian[i][j].type = 'dqidvj'


'''
Function: calculate jacobian elements and update matrix
'''
def calcJacElems(knownnum, jacobian, ybus, t_list, v_list):
    for i in range(knownnum):
        for j in range(knownnum):
            # Ps
            # this i and j isnt from the loop. its from the value from P/Q and V/T
            i_temp = int(jacobian[i][j].name[2])-1
            j_temp = int(jacobian[i][j].name[5])-1
            if jacobian[i][j].type == 'dpidti':
                jacobian[i][j].val = dpidti(i_temp, v_list, ybus, t_list, knownnum)
            elif jacobian[i][j].type == 'dpidtj':
                jacobian[i][j].val = dpidtj(i_temp, j_temp, v_list, ybus, t_list)
            elif jacobian[i][j].type == 'dpidvi':
                jacobian[i][j].val = dpidvi(i_temp, v_list, ybus, t_list, knownnum)
            elif jacobian[i][j].type == 'dpidvj':
                jacobian[i][j].val = dpidvj(i_temp, j_temp, v_list, ybus, t_list)
            elif jacobian[i][j].type == 'dqidti':
                jacobian[i][j].val = dqidti(i_temp, v_list, ybus, t_list, knownnum)
            elif jacobian[i][j].type == 'dqidtj':
                jacobian[i][j].val = dqidtj(i_temp, j_temp, v_list, ybus, t_list)
            elif jacobian[i][j].type == 'dqidvi':
                jacobian[i][j].val = dqidvi(i_temp, v_list, ybus, t_list, knownnum)
            elif jacobian[i][j].type == 'dqidvj':
                jacobian[i][j].val = dqidvj(i_temp, j_temp, v_list, ybus, t_list)
            else:
                print('error')

'''
Function: iterate
should update P, Q, known matrix, unknown matrix, and jacobian
'''
def iterate(knownnum, jacobian, ybus, t_list, v_list, knowns, xmat):
    #first calculate the jacobian matrix
    calcJacElems(knownnum, jacobian, ybus, t_list, v_list)
    #make temp knowns matrix without the names
    new_knowns = [0 for i in range(knownnum)]
    for i in range(knownnum):
        new_knowns[i] = knowns[i].val
    for i in range(knownnum):
        #for each known value, calculate the new value of P or Q and subtract them
        num = int(knowns[i].name[1])-1
        type = knowns[i].name[0]
        if type == 'P':
            #FIXME: P may be negative. will we write this in the excel? or add it in the code?
            new_knowns[i] = new_knowns[i] - calcPVal(num, ybus, knownnum, t_list, v_list)
        else:
            #FIXME: same as P
            new_knowns[i] = new_knowns[i] - calcQVal(num, ybus, knownnum, t_list, v_list)
    #now solve for the new values
    #get temp jac of just values oops
    temp_jac = [[0 for i in range(int(knownnum))] for j in range(int(knownnum))]
    for i in range(knownnum):
        for j in range(knownnum):
            temp_jac[i][j] = jacobian[i][j].val
    new = np.linalg.solve(temp_jac, new_knowns)
    for j in range(knownnum):
        xmat[j].val += new[j]