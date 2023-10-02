# Functions file for Power FLow 4205 Project
from classes import *
import numpy as np
import matplotlib.pyplot as plt

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
"""

def getInitMats(knownNum, xmat, knowns):
    for i in range(knownNum):

        knowns[i].name = input("Enter known #" + str(i + 1) + ": ")
        knowns[i].val = input("Enter associated known value: ")
        numvar = str(knowns[i].name)[1]

        if "p" in knowns[i].name or "P" in knowns[i].name:
            xmat[i].name = "T" + numvar
            knowns[i].name = "P" + numvar
        elif "q" in knowns[i].name or "Q" in knowns[i].name:
            xmat[i].name = "V" + numvar
            knowns[i].name = "Q" + numvar
    return knowns, xmat

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