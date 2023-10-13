# Functions file for Power FLow 4205 Project
from classes import *
import numpy as np
import matplotlib.pyplot as plt
import cmath
"anais comment"
"""
Function: get initial matrices
This function asks the user to input the initial knowns (P, Q) and their associated value.
If the user inputs a P, a T for delta/angle will be added to the unknown matrix. If a Q is
entered, a V will be added.
Parameters:
    knownNum - total number of known Ps and Qs to ask for
    xmat - empty matrix of size busnum rows and 1 column (unknowns matrix)
    knowns - empty matrix of size busnum rows and 1 column (knowns matrix)
Returns:
    known - the filled known matrix with name and values
    xmat - the filled unknown matrix with name only
    pcount - amount of known Ps
    qcount - amount of known Qs
"""

def getInitMats(knownNum, xmat, knowns):
    pcount = 0
    qcount = 0
    for i in range(knownNum):

        knowns[i].name = input("Enter known #" + str(i + 1) + ": ")
        knowns[i].val = input("Enter associated known value: ")
        numvar = str(knowns[i].name)[1]

        if "p" in knowns[i].name or "P" in knowns[i].name:
            xmat[i].name = "T" + numvar
            knowns[i].name = "P" + numvar
            pcount += 1
        elif "q" in knowns[i].name or "Q" in knowns[i].name:
            xmat[i].name = "V" + numvar
            knowns[i].name = "Q" + numvar
            qcount += 1
    return knowns, xmat, pcount, qcount

"""
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
def printMultiMat(num, mat):
    for i in range(int(num)):
        for j in range(int(num)):
            print(mat[i][j].name + ", " + str(mat[i][j].val))

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
def getZYbus(busnum, yBus, zBus):
    for i in range(int(busnum)):
        for j in range(int(busnum)):
            yBus[i][j].name = "Y" + str(i + 1) + str(j + 1)
            # ask for "zbus" values
            if j != i:
                if zBus[j][i] != complex(0, 0) or zBus[j][i] != 0:
                    zBus[i][j] = zBus[j][i]
                else:
                    print("Please enter zero if there is no bus")
                    a = float(input("Enter z" + str(i + 1) + str(j + 1) + " a value: "))
                    b = float(input("Enter z" + str(i + 1) + str(j + 1) + " b value: "))
                    zBus[i][j] = complex(a, b)
    return yBus, zBus

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
    for i in range(num): # row number
        for j in range(num): # column number
            # calc diagnol
            if i == j:
                sum = 0
                for k in range(num):
                    # loop through other numbers and check if its not the same as self
                    # check if impedance i,k isn't zero (avoid inf zero)
                    if k != i and zBus[i][k] != complex(0, 0) and zBus[i][k] != 0:
                        # if true add that value to sum
                        sum += 1 / zBus[i][k]
                if sum == 0: #set to zero
                    yBus[i][j].val = complex(0, 0)
                else: #set to value
                    yBus[i][j].val = sum
            else: #if its not a diagnol element
                #check that its not zero to avoid inf zero
                if zBus[i][j] != complex(0, 0) and zBus[i][j] != 0:
                    yBus[i][j].val = -1 / zBus[i][j]
                else:
                    yBus[i][j].val = complex(0, 0)
    return yBus

#NOT DONE
#FIXME: not sure what im doing with this yes
def getVmat(busnum):
    vMat = [VarMat() for i in range(int(busnum))]
    for j in range(busnum):
        vMat[j].name = "V" + j
        a = float(input("Whats the abs value of V" + j + "?: "))
        #b = float(input("Whats the angle value of V"+ j + "?: "))
        vMat[j].val = a
    return vMat

'''
Function: calculate uij
'''
def uij(gij,bij,thetai,thetaj):
    return (gij*np.sin(thetai-thetaj)-bij*np.cos(thetai-thetaj))

'''
Function: calculate tij
'''
def tij(gij,bij,thetai,thetaj):
    return (gij*np.cos(thetai-thetaj)+bij*np.sin(thetai-thetaj))

'''
Function: calculate P value
'''
def calcPVal(num, yBus, knownNum, T, V):
    p = V[num]^2 * yBus[num][num].val[0]
    for j in range(knownNum):
        p -= V[num] * V[j] * tij(yBus[num][j].val[0], yBus[num][j].val[1], T[num], T[j])
    return p

'''
Function: calculate Q value
'''
def calcQVal(num, yBus, knownNum, T, V):
    q = -V[num]^2 * yBus[num][num].val[1]
    for j in range(knownNum):
        q -= V[num] * V[j] * uij(yBus[num][j].val[0], yBus[num][j].val[1], T[num], T[j])
    return q

'''
Function: calculate partial derivative dPi / dQi
'''
def dpidqi(i, V, yBus, T, knownNum):
    sum = 0
    for j in range(knownNum):
        if j != i:
            sum += V[i] * V[j] * uij(yBus[i][j].val[0], yBus[i][j].val[1], T[i], T[j])
    return sum

'''
Function: calculate partial derivative dPi / dQj
'''
def dpidqj(i, V, yBus, T, j):
    return -V[i] * V[j] * uij(yBus[i][j].val[0], yBus[i][j].val[1], T[i], T[j])

'''
Function: calculate partial derivative dPi / dVi
'''
def dpidvi(i, V, yBus, T, knownnum):
    sum = 2 * V[i] * yBus[i][i].val[0]
    for j in range(knownnum):
        if j != i:
            sum += V[j] * tij(yBus[i][j].val[0], yBus[i][j].val[1], T[i], T[j])
    return sum

'''
Function: calculate partial derivative dPi / dQj
'''
def dpidvj(i, j, V, yBus, T):
    return -V[i]*tij(yBus[i][j].val[0], yBus[i][j].val[1], T[i], T[j])
