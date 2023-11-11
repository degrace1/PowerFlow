# Functions file for Power FLow 4205 Project
from classes import *
import numpy as np
import pandas as pd
import math
import cmath
import time


'''
Function: load file
load initial excel file to get needed lists/info for the rest of the code
'''
def loadFile(filename):
    #read excel file
    initial = pd.read_excel(filename, sheet_name='initial', index_col='bus_num')
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
    q_lim = initial.loc[:, 'q_lim'].to_numpy()


    numP = initial.loc[:, 'P'].count()
    numQ = initial.loc[:, 'Q'].count()
    numT = numP
    numV = numQ
    knownnum = numP + numQ
    line_z = pd.read_excel(filename, sheet_name='line_imp')
    lines = line_z.loc[:, 'line'].to_numpy()
    r_list = line_z.loc[:, 'R'].to_numpy()
    r_list = r_list.astype('float64')
    x_list = line_z.loc[:, 'X'].to_numpy()
    x_list = x_list.astype('float64')
    r_shunt = line_z.loc[:, 'shunt_r'].to_numpy()
    r_shunt = r_shunt.astype('float64')
    x_shunt = line_z.loc[:, 'shunt_x'].to_numpy()
    x_shunt = x_shunt.astype('float64')

    t_x = line_z.loc[:, 't_x'].to_numpy()
    t_x = t_x.astype('float64')
    t_a = line_z.loc[:, 't_a'].to_numpy()
    t_a = t_a.astype('float64')
    return v_list, t_list, p_list, q_list, lines, r_list, x_list, r_shunt, x_shunt, knownnum, busnum, line_z, numT, numV, t_x, t_a, q_lim

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
    newcount = 0
    for j in range(busnum):
        # check if there is a value for initial Q
        # add a V to the xmatrix and a Q to the knowns
        if np.isnan(q_list[j]) == False:
            xmat[newcount + sumcount].name = "V" + str(j + 1)
            knowns[newcount + sumcount].name = "Q" + str(j + 1)
            knowns[newcount + sumcount].val = q_list[j]
            newcount += 1


"""
Function: Set initial guess
This function sets the initial guesses of 1.0pu for voltage and 0 for angle
Parameters:
    knownnum - number of knowns in the system
    xmat - matrix of unknowns with only names
Returns
    xmat - matrix of unknowns with names and initial guesses
"""
def setInitGuess(knownNum, xmat, v_list, t_list):
    for i in range(int(knownNum)):
        if "V" in xmat[i].name:
            xmat[i].val = v_list[int(xmat[i].name[1])-1]
        elif "T" in xmat[i].name:
            xmat[i].val = t_list[int(xmat[i].name[1])-1]


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
        print(xmat[i].name + ": " + str(xmat[i].val))


"""
Function: print multi matrix
This will print a matrix (probably Ybus, jacobian)
Parameters:
    num - number of elem for rows and columns in matrix
    xmat - matrix
Returns:
    nothing, just print
"""
def printMultiMat(num, mat, jac):
    str_print = ['' for i in range(int(num))]
    for i in range(int(num)):
        for j in range(int(num)):
            if jac == False:
                str_print[i] += mat[i][j].name + ": " + str(mat[i][j].val)
            else:
                str_print[i] += mat[i][j].name + ": " + str(mat[i][j].val)
            if j != int(num):
                str_print[i] += ', '
        print(str_print[i])


"""
Function: name Y bus
This will set the names of the Ybus matrix things.

Parameters:
    busnum - number of nuses for matrix size indeces
    yBus - matrix with yBus names and vals
"""

def nameYbus(busnum, yBus):
    for i in range(int(busnum)):
        for j in range(int(busnum)):
            yBus[i][j].name = "Y" + str(i + 1) + str(j + 1)



'''
Function: pi line model
for creating a 2x2 admittance matrix for a one line system. Part 1 of the Cutsem method.
'''
def piLine(r_list, x_list, x_shunt, y_mini, lines, t_x, t_a):
    line_num = lines.size
    #shape: 0 is 11, 1 is 12, 2 is 21, and 3 is 22
    for i in range(line_num):
        if np.isnan(r_list[i]) == False:
            if x_shunt[i] != '' and x_shunt[i] != 0:
                y_mini[i][0] = 1/complex(r_list[i],x_list[i])+(complex(0,x_shunt[i])) #divide by 2 shunt in excel (if whole line)
            else:
                y_mini[i][0] = 1 / complex(r_list[i], x_list[i])
            y_mini[i][1] = -1/complex(r_list[i],x_list[i])
            y_mini[i][2] = y_mini[i][1]
            y_mini[i][3] = y_mini[i][0]
        else:
            y_mini[i][0] = 1/complex(0, t_x[i]) #/(t_a[i]**2)  # turns out this isnt needed
            y_mini[i][1] = -1/complex(0, t_x[i]) #/t_a[i]      # same here
            y_mini[i][2] = y_mini[i][1]
            y_mini[i][3] = 1/complex(0, t_x[i])

'''
Function: Ybus calculations with cutsem algorithm/pi line model
'''
def yBusCutsemCalc(y_mini, lines, yBus):
    for i in range(len(lines)):
        str_temp = str(lines[i])
        a = int(str_temp[0])-1
        b = int(str_temp[1])-1
        yBus[a][a].val += y_mini[i][0]
        yBus[a][b].val += y_mini[i][1]
        yBus[b][a].val += y_mini[i][2]
        yBus[b][b].val += y_mini[i][3]
'''
Function: calculate uij
'''
def uij(gij, bij, thetai, thetaj):
    return (-gij * np.sin(thetai - thetaj) - (-bij * np.cos(thetai - thetaj)))


'''
Function: calculate tij
'''
def tij(gij, bij, thetai, thetaj):
    return (-gij * np.cos(thetai - thetaj) + (-bij * np.sin(thetai - thetaj)))


'''
Function: calculate P value
'''
def calcP(i, yBus, busnum, T, V):
    p = 0
    for j in range(busnum):
        p += V[i]*V[j]*abs(yBus[i][j].val)*np.cos(T[i]-T[j]-cmath.phase(yBus[i][j].val))
    return p

'''
Function: calculate Q value
'''
def calcQ(i, yBus, busnum, T, V):
    q = 0
    for j in range(busnum):
        q += V[i]*V[j]*abs(yBus[i][j].val)*np.sin(T[i]-T[j]-cmath.phase(yBus[i][j].val))
    return q

'''
Function: calculate I in lines
'''
def lineCurrent(line,v_list,t_list,yBus):
    line=str(line)
    i=int(line[0])-1
    j=int(line[1])-1
    I = yBus[i][j].val*(cmath.rect(v_list[i], t_list[i])-cmath.rect(v_list[j], t_list[j]))
    return I

'''
Function: calculate S in lines
'''
def calcSLine(line,v_list,t_list,current):
    line=str(line)
    i=int(line[0])-1
    j=int(line[1])-1
    fromS = cmath.rect(v_list[i], t_list[i])*complex(current.real,-current.imag)
    toS = cmath.rect(v_list[j], t_list[j])*complex(current.real,-current.imag)
    return fromS,toS

'''
Function: calculate P line flows From, To, losses
'''
def calcPFlow(fromS,toS):
    fromP = fromS.real
    toP = toS.real
    pLoss = np.abs(np.abs(toP) - np.abs(fromP))
    return fromP, toP, pLoss
'''
Function: calculate P line flows From, To, losses
'''
def calcqFlow(fromS,toS):
    fromQ = fromS.imag
    toQ= toS.imag
    qLoss = np.abs(np.abs(toQ) - np.abs(fromQ))
    return fromQ, toQ, qLoss

'''
Function: calculate I line flows From, To, losses
'''
def calcIFlow(line,yBus,fromS,toS):
    elem=str(line)
    i=int(line[0])-1
    j=int(line[1])-1
    FromI = cmath.sqrt(np.abs(fromS*yBus[i][j].val))
    ToI= cmath.sqrt(np.abs(toS*yBus[j][i].val))
    return FromI, ToI
"""
Function: print finals
print final values for the power flow method
"""
def printFinals(busnum, p_list, q_list, yBus, t_list, v_list, lines):
    for i in range(busnum):
        if np.isnan(p_list[i]):
            p_list[i] = calcP(i, yBus, busnum, t_list, v_list) # calcPVal(i, yBus, busnum, t_list, v_list)
        if np.isnan(q_list[i]):
            q_list[i] = calcQ(i, yBus, busnum, t_list, v_list) # calcQVal(i, yBus, busnum, t_list, v_list)
    print('Final P and Q Values: ')
    for i in range(busnum):
        print("P", i + 1, ": ", "{:.4f}".format(p_list[i]), "\t\t\t", "Q", i + 1, ": ", "{:.4f}".format(q_list[i]))

    iLine = []
    sLine = []
    pFlow = []
    qFlow = []
    totalPLosses=0
    totalQLosses=0
    for elem in lines:
        line=str(elem)
        current = lineCurrent(line,v_list,t_list,yBus)
        S = calcSLine(line,v_list,t_list,current)
        sLine.append([np.abs(S[0]),np.abs(S[1])])
        P = calcPFlow(S[0],S[1])
        totalPLosses+=P[2]
        pFlow.append([P[0],P[1],P[2]])
        Q = calcqFlow(S[0],S[1])
        totalQLosses+=Q[2]
        qFlow.append([Q[0],Q[1],Q[2]])
        i_current = calcIFlow(line,yBus,S[0],S[1])
        iLine.append([np.abs(i_current[0]),np.abs(i_current[1])])


    print('Flow lines and Losses: ')
    print('P From Bus injection: ', "\t\t",'P To Bus injection: ', "\t\t\t",'P Losses (R.I^2): ')
    for i in range(len(lines)):
        line=str(lines[i])
        print("P", int(line[0]), ": ", "{:.4f}".format(pFlow[i][0]), "\t\t\t", "P", int(line[1]), ": ", "{:.4f}".format(pFlow[i][1]), "\t\t\t", "line ", int(line), ": ", "{:.4f}".format(pFlow[i][2]))
    print('Total P losses = ',"{:.4f}".format(totalPLosses))
    print('Q From Bus injection: ', "\t\t",'Q To Bus injection: ', "\t\t\t",'Q Losses (X.I^2): ')
    for i in range(len(lines)):
        line=str(lines[i])
        print("Q", int(line[0]), ": ", "{:.4f}".format(qFlow[i][0]), "\t\t\t", "Q", int(line[1]), ": ", "{:.4f}".format(qFlow[i][1]), "\t\t\t", "line ", int(line), ": ", "{:.4f}".format(qFlow[i][2]))
    print('Total Q losses = ',"{:.4f}".format(totalQLosses))
    print('Apparent Powers and Currents in lines: ')
    print('S From Bus injection: ', "\t\t",'I From Bus injection: ', "\t\t",'S To Bus injection: ', "\t\t",'I To Bus injection: ')
    for i in range(len(lines)):
        line=str(lines[i])
        print("S", int(line[0]), ": ", "{:.4f}".format(sLine[i][0]), "\t\t\t", "I", int(line[0]),int(line[1]), ": ", "{:.4f}".format(iLine[i][0]), "\t\t\t", "S", int(line[1]), ": ", "{:.4f}".format(sLine[i][1]), "\t\t\t", "I", int(line[1]),int(line[0]), ": ", "{:.4f}".format(iLine[i][1]))

'''
Function: calculate partial derivative dPi / dTi
'''
def dpidti(i, V, yBus, T, busnum):
    i = int(i)
    sum = 0
    for j in range(busnum):
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
def dpidvi(i, V, yBus, T, busnum):
    i = int(i)
    sum = 2 * V[i] * yBus[i][i].val.real
    for j in range(busnum):
        if j != i:
            sum += -V[j] * tij(yBus[i][j].val.real, yBus[i][j].val.imag, T[i], T[j])
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
def dqidti(i, V, yBus, T, busnum):
    i = int(i)
    sum = 0
    for j in range(busnum):
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
def dqidvi(i, V, yBus, T, busnum):
    i = int(i)
    sum = -2 * V[i] * yBus[i][i].val.imag
    for j in range(busnum):
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
def calcJacElems(knownnum, jacobian, ybus, t_list, v_list, busnum):
    for i in range(knownnum):
        for j in range(knownnum):
            # Ps
            # this i and j isnt from the loop. its from the value from P/Q and V/T
            i_temp = int(jacobian[i][j].name[2])-1
            j_temp = int(jacobian[i][j].name[5])-1
            if jacobian[i][j].type == 'dpidti':
                jacobian[i][j].val = dpidti(i_temp, v_list, ybus, t_list, busnum)
            elif jacobian[i][j].type == 'dpidtj':
                jacobian[i][j].val = dpidtj(i_temp, j_temp, v_list, ybus, t_list)
            elif jacobian[i][j].type == 'dpidvi':
                jacobian[i][j].val = dpidvi(i_temp, v_list, ybus, t_list, busnum)
            elif jacobian[i][j].type == 'dpidvj':
                jacobian[i][j].val = dpidvj(i_temp, j_temp, v_list, ybus, t_list)
            elif jacobian[i][j].type == 'dqidti':
                jacobian[i][j].val = dqidti(i_temp, v_list, ybus, t_list, busnum)
            elif jacobian[i][j].type == 'dqidtj':
                jacobian[i][j].val = dqidtj(i_temp, j_temp, v_list, ybus, t_list)
            elif jacobian[i][j].type == 'dqidvi':
                jacobian[i][j].val = dqidvi(i_temp, v_list, ybus, t_list, busnum)
            elif jacobian[i][j].type == 'dqidvj':
                jacobian[i][j].val = dqidvj(i_temp, j_temp, v_list, ybus, t_list)
            else:
                print('error')

'''
Function: calculate jacobian elements and update matrix for DLF
'''
def calcJacElemsDLF(knownnum, jacobian, ybus, t_list, v_list, busnum):
    for i in range(knownnum):
        for j in range(knownnum):
            # Ps
            # this i and j isnt from the loop. its from the value from P/Q and V/T
            i_temp = int(jacobian[i][j].name[2])-1
            j_temp = int(jacobian[i][j].name[5])-1
            if jacobian[i][j].type == 'dpidti':
                jacobian[i][j].val = dpidti(i_temp, v_list, ybus, t_list, busnum)
            elif jacobian[i][j].type == 'dpidtj':
                jacobian[i][j].val = dpidtj(i_temp, j_temp, v_list, ybus, t_list)
            elif jacobian[i][j].type == 'dpidvi':
                jacobian[i][j].val = 0
            elif jacobian[i][j].type == 'dpidvj':
                jacobian[i][j].val = 0
            elif jacobian[i][j].type == 'dqidti':
                jacobian[i][j].val = 0
            elif jacobian[i][j].type == 'dqidtj':
                jacobian[i][j].val = 0
            elif jacobian[i][j].type == 'dqidvi':
                jacobian[i][j].val = dqidvi(i_temp, v_list, ybus, t_list, busnum)
            elif jacobian[i][j].type == 'dqidvj':
                jacobian[i][j].val = dqidvj(i_temp, j_temp, v_list, ybus, t_list)
            else:
                print('error')

'''
Function: iterate
should update P, Q, known matrix, unknown matrix, and jacobian
'''
def iterate(knownnum, jacobian, ybus, t_list, v_list, knowns, xmat, busnum, qnum, num_lims, type):
    #first calculate the jacobian matrix
    if type == 'NR':
        calcJacElems(knownnum, jacobian, ybus, t_list, v_list, busnum)
    elif type == 'DLF':
        calcJacElemsDLF(knownnum, jacobian, ybus, t_list, v_list, busnum)
    else:
        print('error thrown in which type of jacobian calculation')
    #make temp knowns matrix without the names
    new_knowns = [0 for i in range(knownnum)]
    net_injections = [0 for i in range(knownnum)]
    for i in range(knownnum):
        new_knowns[i] = knowns[i].val
    for i in range(knownnum):
        #for each known value, calculate the new value of P or Q and subtract them
        num = int(knowns[i].name[1])-1
        type = knowns[i].name[0]
        if type == 'P':
            #Note: change generating/not +/- for P and Q IN EXCEL
            new_p = calcP(num, ybus, busnum, t_list, v_list) # calcPVal(num, ybus, busnum, t_list, v_list)
            net_injections[i] = new_p
            new_knowns[i] = new_knowns[i] - new_p
        else:
            #Note: change generating/not +/- for P and Q IN EXCEL
            new_q = calcQ(num, ybus, busnum, t_list, v_list) # calcQVal(num, ybus, busnum, t_list, v_list)
            net_injections[i] = new_q
            new_knowns[i] = new_knowns[i] - new_q
    print("Net Injections: ")
    for i in range(knownnum):
        print(knowns[i].name, ': ', net_injections[i])


    temp_jac = [[0 for i in range(int(knownnum))] for j in range(int(knownnum))]
    for i in range(knownnum):
        for j in range(knownnum):
            temp_jac[i][j] = jacobian[i][j].val
    corrections = np.linalg.solve(temp_jac, new_knowns)
    for j in range(knownnum):
        xmat[j].val += corrections[j]
        temp_num = int(xmat[j].name[1])
        temp_val = xmat[j].val
        if xmat[j].name[0] == "T":
            t_list[(temp_num-1)] = temp_val
        elif xmat[j].name[0] == "V":
            v_list[(temp_num-1)] = temp_val
        else:
            print("error thrown in updating v and t lists")
    if num_lims>=1:
        q_limit = [0 for i in range(num_lims)]
        for i in range(num_lims):
            q_limit[i] = calcQ(qnum[i]-1, ybus, busnum, t_list, v_list)
    else:
        q_limit = 0

    print("Jacobian: ")
    printMultiMat(knownnum, jacobian, True)
    print("Corrections Vector (dV, dT): ")
    print(corrections)
    print("RHS Vector (dP, dQ): ")
    print(new_knowns)
    print("New voltage magnitudes and angles (in degrees):")
    for i in range(busnum):
        print("|V|", i + 1, ": ", "{:.4f}".format(v_list[i]), "\t\t", "Theta", i + 1, ": ",
              "{:.4f}".format(t_list[i] * 180 / math.pi))

    return corrections, q_limit

def loop_normal(knowns, knownnum, jacobian, yBus, t_list, v_list, xmat, busnum, conv_crit, type):
    convergence = False
    itno = 0
    while not (convergence or itno > 10):
        itno += 1
        print("\n\nIteration #" + str(itno))
        outputs = iterate(knownnum, jacobian, yBus, t_list, v_list, knowns, xmat, busnum, 0, 0, type)
        corrections = outputs[0]
        # Check corrections matrix for convergence
        count = 0
        for i in range(corrections.size):
            if abs(corrections[i]) > conv_crit:
                cur = abs(corrections[i])
                count += 1
        convergence = count == 0


'''
Function: NR iterate loop with Q limits
this function should loop through iterations of newton rhapson (or can be other if we change the jacobian method).
Input parameters:
    - knowns - mismatch vector of knowns
    - knownum - number of Ps and Qs in knowns
    - yBus - ybus
    - t_list - list of voltage angles
    - v_list - list of voltage magnitudes
    - xmat - matrix of unkowns (angles and magnitudes)
    - busnum - total bus number
    - conv_crit - convergence criteria value
    - p_list - list of active powers per bus
    - q_list - list of reactive powers per bus
    - q_lim - ARRAY of limit values (same length as q_list, but empty when theres no limit)
    - num_p
'''

def NR_iterate_loop_qlim(knowns, knownnum,yBus, t_list, v_list, xmat, busnum, conv_crit, p_list, q_list, q_lim, num_p):
    convergence = False
    itno = 0
    P = np.zeros(busnum)
    Q = np.zeros(busnum)
    #remember original V values
    v_orig = [1 for i in range(busnum)]  # original set magnitude of voltage
    for i in range(busnum):
        v_orig[i] = v_list[i]
    #create copies of xmatrix and knowns
    new_xmat = [VarMat() for i in range(int(knownnum))]
    new_knowns = [VarMat() for i in range(int(knownnum))]
    for i in range(knownnum):
        new_xmat[i].name = xmat[i].name
        new_xmat[i].val = xmat[i].val
        new_knowns[i].name = knowns[i].name
        new_knowns[i].val = knowns[i].val

    i_qlim_lower = []
    i_qlim_upper = []
    while not (convergence or itno>10):
        print("iteration numer: " + str(itno))
        #calc all P/Q
        for i in range(busnum):
            P[i] = calcP(i, yBus, busnum, t_list, v_list)
            Q[i] = calcQ(i, yBus, busnum, t_list, v_list)

        # check the new Q values and add their indices to a list if the limit is violated
        # set the q_list values to the lower or upper limit
        count_lims_reached = 0
        for i in range(len(q_list)):
            if (i not in i_qlim_lower and i not in i_qlim_upper):
                if not np.isnan(q_lim[i]) and Q[i] < -q_lim[i]:
                    q_list[i] = -q_lim[i]
                    i_qlim_lower.append(i)
                    count_lims_reached+=1
                    knownnum += 1
                elif not np.isnan(q_lim[i]) and Q[i] > q_lim[i]:
                    q_list[i] = q_lim[i]
                    i_qlim_upper.append(i)
                    count_lims_reached+=1
                    knownnum += 1

        #only add on to matricess if the count is greater than zero
        if count_lims_reached > 0:
            print("Q limit reached, switching to PQ bus: ")
            for i in i_qlim_lower:
                known_cur1 = VarMat('Q' + str(i+1), q_list[i])
                new_knowns.insert(i+num_p, known_cur1)
                xmat_cur1 = VarMat('V'+str(i+1), 1)
                new_xmat.insert(i+num_p, xmat_cur1)

            for i in i_qlim_upper:
                known_cur1 = VarMat('Q' + str(i + 1), q_list[i])
                new_knowns.insert(i + num_p, known_cur1)
                #strcur = 'V' + str(i+1)
                xmat_cur2 = VarMat('V' + str(i + 1), 1)
                new_xmat.insert(i + num_p, xmat_cur2)


        #establish jacobian matrix
        new_jacobian = [[JacElem() for i in range(int(knownnum))] for j in range(int(knownnum))]
        nameJacElem(knownnum, new_knowns, new_xmat, new_jacobian)
        calcJacElems(knownnum, new_jacobian, yBus, t_list, v_list, busnum)

        #create temp jac without the object and just numbers
        temp_jac = [[0 for i in range(int(knownnum))] for j in range(int(knownnum))]
        #create temp knowns without the object and just numbers
        temp_knowns = np.zeros(knownnum)
        for i in range(knownnum):
            temp_knowns[i] = new_knowns[i].val
            for j in range(knownnum):
                temp_jac[i][j] = new_jacobian[i][j].val

        net_injections = np.zeros(knownnum)

        for i in range(knownnum):
            # for each known value, calculate the new value of P or Q and subtract them
            num = int(new_knowns[i].name[1]) - 1
            type = new_knowns[i].name[0]
            if type == 'P':
                # Note: change generating/not +/- for P and Q IN EXCEL
                new_p = P[num]
                net_injections[i] = new_p
                temp_knowns[i] = temp_knowns[i] - new_p
            else:
                # Note: change generating/not +/- for P and Q IN EXCEL
                new_q = Q[num]
                net_injections[i] = new_q
                temp_knowns[i] = temp_knowns[i] - new_q

        print("Net Injections: ")
        for i in range(knownnum):
            print(new_knowns[i].name, ': ', net_injections[i])

        #solve the linear algebra
        corrections = np.linalg.solve(temp_jac, temp_knowns)
        #update the magnitudes and angles
        for j in range(knownnum):
            new_xmat[j].val += corrections[j]
            temp_num = int(new_xmat[j].name[1])
            temp_val = new_xmat[j].val
            if new_xmat[j].name[0] == "T":
                t_list[(temp_num - 1)] = temp_val
            elif new_xmat[j].name[0] == "V":
                v_list[(temp_num - 1)] = temp_val
            else:
                print("error thrown in updating v and t lists")

        #one loop is done, now we check to see if the reactive power limits are still violated
        #check the upper limit violations
        for i in i_qlim_upper:
            if v_list[i] <= v_orig[i]:
                q_list[i] = q_lim[i]
            elif v_list[i] > v_orig[i] and Q[i] >= q_lim[i]:
                q_list[i] = q_lim[i]
            elif v_list[i] > v_orig[i] and Q[i] <= -q_lim[i]:
                q_list[i] = -q_lim[i]
            else:
                # set v value back to original
                v_list[i] = v_orig[i]
                i_del = i
                # delete from upper list
                i_qlim_upper = [x for x in i_qlim_upper if x != i_del]
                # set q back to nan
                q_list[i] = np.nan
                # delete from the xmatrix and mismatch
                np.delete(new_xmat, i)
                np.delete(new_knowns, i)
                knownnum -= 1

        #check the lower limit violations
        for i in i_qlim_lower:
            if v_list[i] >= v_orig[i]:
                q_list[i] = -q_lim[i]
            elif v_list[i] < v_orig[i] and Q[i] <= -q_lim[i]:
                q_list[i] = -q_lim[i]
            elif v_list[i] < v_orig[i] and Q[i] >= q_lim[i]:
                q_list[i] = -q_lim[i]
            else:
                # set v back to original
                v_list[i] = v_orig[i]
                i_del = i
                # delete from lower list
                i_qlim_lower = [x for x in i_qlim_lower if x != i_del]
                # set q back to nan
                q_list[i] = np.nan
                # delete from xmatrix and mismatch
                np.delete(new_xmat, i)
                np.delete(new_knowns, i)
                knownnum -= 1

        print("Corrections Vector (dV, dT): ")
        print(corrections)
        print("RHS Vector (dP, dQ): ")
        print(temp_knowns)
        print("New voltage magnitudes and angles (in degrees):")
        for i in range(busnum):
            print("|V|", i + 1, ": ", "{:.4f}".format(v_list[i]), "\t\t", "Theta", i + 1, ": ",
                  "{:.4f}".format(t_list[i] * 180 / math.pi))

        # Check corrections matrix for convergence
        count = 0
        for i in range(corrections.size):
            if abs(corrections[i]) > conv_crit:
                cur = abs(corrections[i])
                count += 1
        convergence = count == 0
        itno += 1

def newtonRhapson(conv_crit, qlimType, filename):
    startNRTime = time.time()
    stuff = loadFile(filename)
    v_list = stuff[0]
    t_list = stuff[1]
    p_list = stuff[2]
    q_list = stuff[3]
    lines = stuff[4]
    r_list = stuff[5]
    x_list = stuff[6]
    r_shunt = stuff[7]
    x_shunt = stuff[8]
    knownnum = stuff[9]
    busnum = stuff[10]
    line_z = stuff[11]
    numT = stuff[12]
    numV = stuff[13]
    t_x = stuff[14]
    t_a = stuff[15]
    q_lim = stuff[16]

    knowns = [VarMat() for i in range(int(knownnum))]
    xmat = [VarMat() for j in range(int(knownnum))]


    getInitMats(xmat, knowns, p_list, q_list, busnum)
    setInitGuess(knownnum, xmat, v_list, t_list)


    yBus = [[VarMat() for i in range(int(busnum))] for j in range(int(busnum))]

    nameYbus(busnum, yBus)


    y_mini = [[complex(0, 0) for i in range(4)] for j in range(lines.size)]
    piLine(r_list, x_list, x_shunt, y_mini, lines, t_x, t_a)
    yBusCutsemCalc(y_mini, lines, yBus)
    #printMultiMat(busnum, yBus, False)
    jacobian = [[JacElem() for i in range(int(knownnum))] for j in range(int(knownnum))]
    nameJacElem(knownnum, knowns, xmat, jacobian)
    if qlimType == 'none':
        loop_normal(knowns, knownnum, jacobian, yBus, t_list, v_list, xmat, busnum, conv_crit, 'NR')
    elif qlimType == 'each':
        num_p = numT
        NR_iterate_loop_qlim(knowns, knownnum, yBus, t_list, v_list, xmat, busnum, conv_crit, p_list, q_list, q_lim,
                             num_p)
    else:
        print("error thrown in deciding reactive power limit iteration method, retype qlimtype")

    printFinals(busnum, p_list, q_list, yBus, t_list, v_list, lines)
    endTimeNR = time.time()
    print("Final running time for Newton-Raphson method is: ", endTimeNR-startNRTime, "seconds")

'''
Function: Calculate DC Power Flow
'''
def calcDCPF(filenameDCPF):
    stuff = loadFile(filenameDCPF)
    p_list = stuff[2]
    lines = stuff[4]
    r_list = stuff[5]
    x_list = stuff[6]
    x_shunt = stuff[8]
    knownnum = stuff[9]
    busnum = stuff[10]
    line_z = stuff[11]
    t_x = stuff[14]
    t_a = stuff[15]

    #Find slack_bus

    #Remove row and column corresponding to slack bus
    ##check excel for bus_type and get index=slack_bus
    initial= pd.read_excel(filenameDCPF, sheet_name='initial')
    bus_type=initial.loc[:,'bus_type']
    slack_bus=np.where(bus_type=='slack')[0][0]

    # Obtain the yBus
    yBus = [[VarMat() for i in range(int(busnum))] for j in range(int(busnum))]

    nameYbus(busnum, yBus)
    y_mini = [[complex(0, 0) for i in range(4)] for j in range(lines.size)]
    piLine(r_list, x_list, x_shunt, y_mini, lines, t_x, t_a)
    yBusCutsemCalc(y_mini, lines, yBus)

    #Remove slack bus from Ybus and neglect the real part. Remove j from imaginary part
    yBus_wo_slack=np.empty([len(yBus)-1, len(yBus)-1])
    new_i = 0
    for i in range(busnum):
        if i == slack_bus:
            continue  # Skip the slack bus row
        new_j = 0
        for j in range(busnum):
            if j == slack_bus:
                continue  # Skip the slack bus column
            yBus_wo_slack[new_i][new_j] = yBus[i][j].val.imag
            new_j += 1
        new_i += 1

    #Multiply the matrix by -1
    yBusDC=-1*yBus_wo_slack

    p_without_slack = np.delete(p_list, slack_bus, axis=0)#delete slack bus
    p_without_slack[np.isnan(p_without_slack)] = 0
    xmat=np.linalg.solve(yBusDC,p_without_slack)
    new_col = np.array([[0]])
    xmat_final = np.insert(xmat, slack_bus, new_col, axis=0)

    #calculate P line flows
    slack_bus = int(slack_bus)
    sum = 0
    PlineDC = []
    for line in lines:
        try:
            i, j = divmod(line, 10)
            i -= 1  # Convert to 0-based
            j -= 1  # Convert to 0-based
            value = abs(yBus[i][j].val.imag) * (xmat_final[i] - xmat_final[j])
            PlineDC.append(value)
        except (ValueError, IndexError):
            PlineDC.append(None)

    return slack_bus+1, xmat_final, PlineDC

'''
Function: Prints DC Power Flow
'''
def printDCPF(filenameDCPF):
    startTimeDC = time.time()
    #Read the excel and save variables
    stuff = loadFile(filenameDCPF)
    knownnum = stuff[9]
    busnum = stuff[10]
    knowns = [VarMat() for i in range(int(knownnum))]
    yBus = [[VarMat() for i in range(int(busnum))] for j in range(int(busnum))]

    DCPF=calcDCPF(filenameDCPF)
    print('Slack bus:', DCPF[0])
    print('Angles (º):', DCPF[1])
    print('Active Power flow through lines (pu):', DCPF[2])
    endTimeDC = time.time()
    print("Final running time for DC method is: ", endTimeDC - startTimeDC, "seconds")

'''
Function: loadbustype
gets a string array with the bus types
'''
def loadbustype(filenameFDLF):
    #read excel file
    initial = pd.read_excel(filenameFDLF, sheet_name='initial', index_col='bus_num')
    #extract bus type as string
    type_list = initial.loc[:, 'bus_type'].to_numpy()
    type_list = np.array(type_list)
    type_list = type_list.astype('str')
    return type_list

'''
Function: iterate FDLF
performs one iteration of fast decoupled load flow
'''
def iterate_FDLF(knownnum, ybus, t_list, v_list, knowns, xmat, busnum, bp_inv, bpp_inv, slackbus, pvbus, notype, type_it, pvBusNo):
    #make temp knowns matrix without the names
    new_knowns = [0 for i in range(knownnum)]
    net_injections = [0 for i in range(knownnum)]
    for i in range(knownnum):
        new_knowns[i] = knowns[i].val
    dP_V = [0 for i in range(busnum - len(slackbus) - len(notype))]
    lenPVbus = len(pvbus) - pvBusNo
    dQ_V = [0 for i in range(busnum - len(slackbus) - lenPVbus - len(notype))]
    corrections = [0 for i in range(knownnum)]
    cueP = 0
    cueQ = 0

    if type_it == 'end_it':
        # calculate P and Q mismatches
        for i in range(knownnum):
            #for each known value, calculate the new value of P or Q and subtract them
            num = int(knowns[i].name[1])-1
            type = knowns[i].name[0]
            if type == 'P':
                #Note: change generating/not +/- for P and Q IN EXCEL
                new_p = calcP(num, ybus, busnum, t_list, v_list)
                net_injections[i] = new_p
                new_knowns[i] = new_knowns[i] - new_p
                dP_V[cueP] = new_knowns[i] / v_list[num]
                cueP += 1
            else:
                #Note: change generating/not +/- for P and Q IN EXCEL
                new_q = calcQ(num, ybus, busnum, t_list, v_list)
                net_injections[i] = new_q
                new_knowns[i] = new_knowns[i] - new_q
                dQ_V[cueQ] = new_knowns[i] / v_list[num]
                cueQ += 1
        # solve for dT and dV
        dT = -np.dot(bp_inv, np.transpose(dP_V))
        dV = -np.dot(bpp_inv, np.transpose(dQ_V))
        cueP = 0
        cueQ = 0
        for j in range(knownnum):
            if xmat[j].name[0] == "T":
                corrections[j] = dT[cueP]
                xmat[j].val += corrections[j]
                t_list[(int(xmat[j].name[1]) - 1)] = xmat[j].val
                cueP += 1
            elif xmat[j].name[0] == "V":
                corrections[j] = dV[cueQ]
                xmat[j].val += corrections[j]
                v_list[(int(xmat[j].name[1]) - 1)] = xmat[j].val
                cueQ += 1
            else:
                print("error thrown in updating v and t lists")

    elif type_it == 'mid_it':
        # calculate P mismatches
        for i in range(knownnum):
            #for each known value, calculate the new value of P or Q and subtract them
            num = int(knowns[i].name[1])-1
            type = knowns[i].name[0]
            if type == 'P':
                #Note: change generating/not +/- for P and Q IN EXCEL
                new_p = calcP(num, ybus, busnum, t_list, v_list)
                net_injections[i] = new_p
                new_knowns[i] = new_knowns[i] - new_p
                dP_V[cueP] = new_knowns[i] / v_list[num]
                cueP += 1
        # solve for dT
        dT = -np.dot(bp_inv, np.transpose(dP_V))
        # update the angles
        cueP = 0
        for j in range(knownnum):
            if xmat[j].name[0] == "T":
                corrections[j] = dT[cueP]
                xmat[j].val += corrections[j]
                t_list[(int(xmat[j].name[1]) - 1)] = xmat[j].val
                cueP += 1
        # calculate Q mismatches
        for i in range(knownnum):
            # for each known value, calculate the new value of P or Q and subtract them
            num = int(knowns[i].name[1]) - 1
            type = knowns[i].name[0]
            if type == 'Q':
                # Note: change generating/not +/- for Q and Q IN EXCEL
                new_q = calcQ(num, ybus, busnum, t_list, v_list)
                net_injections[i] = new_q
                new_knowns[i] = new_knowns[i] - new_q
                dQ_V[cueQ] = new_knowns[i] / v_list[num]
                cueQ += 1
        # solve for dV
        dV = -np.dot(bpp_inv, np.transpose(dQ_V))
        # update the voltages
        cueQ = 0
        for j in range(knownnum):
            if xmat[j].name[0] == "V":
                corrections[j] = dV[cueQ]
                xmat[j].val += corrections[j]
                v_list[(int(xmat[j].name[1]) - 1)] = xmat[j].val
                cueQ += 1
    else:
        print("error in type of iteration (end_it or mid_it)")

    print("Net Injections: ")
    for i in range(knownnum):
        print(knowns[i].name, ': ', net_injections[i])

    print("Corrections Vector (dV, dT): ")
    print(corrections)
    print("RHS Vector (dP, dQ): ")
    print(new_knowns)
    print("New voltage magnitudes and angles (in degrees):")
    for i in range(busnum):
        print("|V|", i + 1, ": ", "{:.4f}".format(v_list[i]), "\t\t", "Theta", i + 1, ": ",
              "{:.4f}".format(t_list[i] * 180 / math.pi))
    return corrections

def iterate_fdlf_qlim(conv_crit, busnum, knownnum, slackbus, pvbus, notype, knowns, xmat, yBus, t_list, v_list, q_list, q_lim, num_p, type_it):
    i_qlim_upper=[]
    i_qlim_lower=[]

    #copy original v values to remember them
    v_orig = [1 for i in range(busnum)]  # original set magnitude of voltage
    for i in range(busnum):
        v_orig[i] = v_list[i]

    new_xmat = [VarMat() for i in range(int(knownnum))]
    new_knowns = [VarMat() for i in range(int(knownnum))]
    for i in range(knownnum):
        new_xmat[i].name = xmat[i].name
        new_xmat[i].val = xmat[i].val
        new_knowns[i].name = knowns[i].name
        new_knowns[i].val = knowns[i].val

    convergence = False
    itno = 0
    while not (convergence or itno > 10):
        itno += 1
        print("\n\nIteration #" + str(itno))

    # when we add a Q to the knowns, there will be one less PV bus to remove from bpp and dQ_V

        # CaLlculate new P and Q values
        P = np.zeros(busnum)
        Q = np.zeros(busnum)
        for i in range(busnum):
            P[i] = calcP(i, yBus, busnum, t_list, v_list)
            Q[i] = calcQ(i, yBus, busnum, t_list, v_list)

        count_lims_reached = 0
        for i in range(len(q_list)):
            if (i not in i_qlim_lower and i not in i_qlim_upper):
                if not np.isnan(q_lim[i]) and Q[i] < -q_lim[i]:
                    q_list[i] = -q_lim[i]
                    i_qlim_lower.append(i)
                    count_lims_reached += 1
                    knownnum += 1
                elif not np.isnan(q_lim[i]) and Q[i] > q_lim[i]:
                    q_list[i] = q_lim[i]
                    i_qlim_upper.append(i)
                    count_lims_reached += 1
                    knownnum += 1

        # only add on to matrices if the count is greater than zero
        if count_lims_reached > 0:
            print("Switching to PQ bus: ")
            for i in i_qlim_lower:
                known_cur1 = VarMat('Q' + str(i + 1), q_list[i])
                new_knowns.insert(i + num_p, known_cur1)
                xmat_cur1 = VarMat('V' + str(i + 1), 1)
                new_xmat.insert(i + num_p, xmat_cur1)

            for i in i_qlim_upper:
                known_cur1 = VarMat('Q' + str(i + 1), q_list[i])
                new_knowns.insert(i + num_p, known_cur1)
                # strcur = 'V' + str(i+1)
                xmat_cur2 = VarMat('V' + str(i + 1), 1)
                new_xmat.insert(i + num_p, xmat_cur2)

        #calculate new bp vals based on how many Qs have passed limits
        bp_inv, bpp_inv = fdlfBprimes(busnum, yBus, slackbus, pvbus, notype, i_qlim_lower, i_qlim_upper)

        total_lims_reached = len(i_qlim_lower) + len(i_qlim_upper)
        corrections = iterate_FDLF(knownnum, yBus, t_list, v_list, new_knowns, new_xmat, busnum, bp_inv, bpp_inv, slackbus, pvbus, notype, type_it, total_lims_reached)

        #one loop is done, now we check to see if the reactive power limits are still violated
        #check the upper limit violations
        for i in i_qlim_upper:
            if v_list[i] <= v_orig[i]:
                q_list[i] = q_lim[i]
            elif v_list[i] > v_orig[i] and Q[i] >= q_lim[i]:
                q_list[i] = q_lim[i]
            elif v_list[i] > v_orig[i] and Q[i] <= -q_lim[i]:
                q_list[i] = -q_lim[i]
            else:
                # set v value back to original
                v_list[i] = v_orig[i]
                i_del = i
                # delete from upper list
                i_qlim_upper = [x for x in i_qlim_upper if x != i_del]
                # set q back to nan
                q_list[i] = np.nan
                # delete from the xmatrix and mismatch
                np.delete(new_xmat, i)
                np.delete(new_knowns, i)
                knownnum -= 1

        #check the lower limit violations
        for i in i_qlim_lower:
            if v_list[i] >= v_orig[i]:
                q_list[i] = -q_lim[i]
            elif v_list[i] < v_orig[i] and Q[i] <= -q_lim[i]:
                q_list[i] = -q_lim[i]
            elif v_list[i] < v_orig[i] and Q[i] >= q_lim[i]:
                q_list[i] = -q_lim[i]
            else:
                # set v back to original
                v_list[i] = v_orig[i]
                i_del = i
                # delete from lower list
                i_qlim_lower = [x for x in i_qlim_lower if x != i_del]
                # set q back to nan
                q_list[i] = np.nan
                # delete from xmatrix and mismatch
                np.delete(new_xmat, i)
                np.delete(new_knowns, i)
                knownnum -= 1

        # Check corrections matrix for convergence
        count = 0
        for i in range(knownnum):
            if abs(corrections[i]) > conv_crit:
                cur = abs(corrections[i])
                count += 1
        convergence = count == 0





"""
Function: normal loop for fast decoupled load flow
iterates until convergence is met or the loop goes 10 times
"""
def loop_normal_FDLF(knowns, knownnum, yBus, t_list, v_list, xmat, busnum, conv_crit, bp_inv, bpp_inv, slackbus, pvbus, notype, type_it, qlim_type, q_list, q_lim, num_p):

    convergence = False
    itno = 0
    while not (convergence or itno > 10):
        itno += 1
        print("\n\nIteration #" + str(itno))

        corrections = iterate_FDLF(knownnum, yBus, t_list, v_list, knowns, xmat, busnum, bp_inv, bpp_inv, slackbus, pvbus, notype, type_it, 0)
        # Check corrections matrix for convergence
        count = 0
        for i in range(knownnum):
            if abs(corrections[i]) > conv_crit:
                cur = abs(corrections[i])
                count += 1
        convergence = count == 0

"""
FunctionL fast deoucpled load flow b primes creation
makes the inverse B' and B'' for use in calculations
Parameters:
    - filename - str for excel file
    - busnum - int total number of buses
    - yBus - matrix impedance values
Returns:
    - bp_inv - inverse matrix of B'
    - bpp_inv - inverse matrix of B''
"""
def fdlfBprimes(busnum, yBus, slackbus, pvbus, notype, i_qlim_lower, i_qlim_upper):

    for j in range(len(i_qlim_lower)):
        if i_qlim_lower[j] in pvbus:
            indexcur = np.where(pvbus==i_qlim_lower[j])
            pvbus = np.delete(pvbus, indexcur)

    for j in range(len(i_qlim_upper)):
        if i_qlim_upper[j] in pvbus:
            indexcur = np.where(pvbus==i_qlim_upper[j])
            pvbus = np.delete(pvbus, indexcur)


    # Obtain the inverses of B' and B''
    bp = np.empty([busnum - len(slackbus) - len(notype), busnum - len(slackbus) - len(notype)])
    bp[:] = np.nan
    bpp = np.empty(
        [busnum - len(slackbus) - len(pvbus) - len(notype), busnum - len(slackbus) - len(pvbus) - len(notype)])
    bpp[:] = np.nan
    ic = 0
    jc = 0
    icc = 0
    jcc = 0
    for i in range(busnum):
        for j in range(busnum):
            if i not in slackbus and j not in slackbus and i not in notype and j not in notype:
                bp[ic, jc] = yBus[i][j].val.imag
                if jc < busnum - len(slackbus) - len(notype) - 1:
                    jc += 1
                else:
                    jc = 0
                    ic += 1
                if i not in pvbus and j not in pvbus:
                    bpp[icc, jcc] = yBus[i][j].val.imag
                    if jcc < busnum - len(slackbus) - len(pvbus) - len(notype) - 1:
                        jcc += 1
                    else:
                        jcc = 0
                        icc += 1
    bp_inv = np.linalg.inv(bp)
    bpp_inv = np.linalg.inv(bpp)
    return bp_inv, bpp_inv

'''
Function: FastDecoupled
performs Fast Decoupled load flow method
'''
def fastDLF(conv_crit, qlimType, filename, it_type):
    startTimeFast = time.time()
    stuff = loadFile(filename)
    v_list = stuff[0]
    t_list = stuff[1]
    p_list = stuff[2]
    q_list = stuff[3]
    lines = stuff[4]
    r_list = stuff[5]
    x_list = stuff[6]
    r_shunt = stuff[7]
    x_shunt = stuff[8]
    knownnum = stuff[9]
    busnum = stuff[10]
    line_z = stuff[11]
    numT = stuff[12]
    numV = stuff[13]
    t_x = stuff[14]
    t_a = stuff[15]
    q_lim = stuff[16]

    knowns = [VarMat() for i in range(int(knownnum))]
    xmat = [VarMat() for j in range(int(knownnum))]
    getInitMats(xmat, knowns, p_list, q_list, busnum)
    setInitGuess(knownnum, xmat, v_list, t_list)
    yBus = [[VarMat() for i in range(int(busnum))] for j in range(int(busnum))]
    nameYbus(busnum, yBus)
    y_mini = [[complex(0, 0) for i in range(4)] for j in range(lines.size)]
    piLine(r_list, x_list, x_shunt, y_mini, lines, t_x, t_a)
    yBusCutsemCalc(y_mini, lines, yBus)

    # Get slack and pv buses
    type_list = loadbustype(filename)
    slackbus = np.where(type_list == 'slack')[0]
    pvbus = np.where(type_list == 'pv')[0]
    notype = np.where(type_list == 'nan')[0]

    #get the B' and B'' inverses
    bp_inv, bpp_inv = fdlfBprimes(busnum, yBus, slackbus, pvbus, notype, [], [])

    if qlimType == 'none':
        loop_normal_FDLF(knowns, knownnum, yBus, t_list, v_list, xmat, busnum, conv_crit, bp_inv, bpp_inv, slackbus, pvbus, notype, it_type, qlimType, q_list, q_lim, numT)
    elif qlimType == 'each':
        iterate_fdlf_qlim(conv_crit, busnum, knownnum, slackbus, pvbus, notype, knowns, xmat, yBus, t_list, v_list, q_list, q_lim, numT, it_type)
    else:
        print("error in picking qlim type for fast decoupled")

    printFinals(busnum, p_list, q_list, yBus, t_list, v_list, lines)
    endTimeFast = time.time()
    print("Final running time for Fast Decoupled method is: ", endTimeFast-startTimeFast, "seconds")



'''
Function: decoupledLoadFlow
algorithm with the Decoupled Load Flow method
'''
def decoupledLoadFlow(conv_crit, filenameDLF):
    startTimeDecoup = time.time()
    stuff = loadFile(filenameDLF)
    v_list = stuff[0]
    t_list = stuff[1]
    p_list = stuff[2]
    q_list = stuff[3]
    lines = stuff[4]
    r_list = stuff[5]
    x_list = stuff[6]
    r_shunt = stuff[7]
    x_shunt = stuff[8]
    knownnum = stuff[9]
    busnum = stuff[10]
    line_z = stuff[11]
    numT = stuff[12]
    numV = stuff[13]
    t_x = stuff[14]
    t_a = stuff[15]

    knowns = [VarMat() for i in range(int(knownnum))]
    xmat = [VarMat() for j in range(int(knownnum))]


    getInitMats(xmat, knowns, p_list, q_list, busnum)
    setInitGuess(knownnum, xmat, v_list, t_list)


    yBus = [[VarMat() for i in range(int(busnum))] for j in range(int(busnum))]
    nameYbus(busnum, yBus)


    y_mini = [[complex(0, 0) for i in range(4)] for j in range(lines.size)]
    piLine(r_list, x_list, x_shunt, y_mini, lines, t_x, t_a)
    yBusCutsemCalc(y_mini, lines, yBus)

    jacobian = [[JacElem() for i in range(int(knownnum))] for j in range(int(knownnum))]
    nameJacElem(knownnum, knowns, xmat, jacobian)
    loop_normal(knowns, knownnum, jacobian, yBus, t_list, v_list, xmat, busnum, conv_crit, 'DLF')

    printFinals(busnum, p_list, q_list, yBus, t_list, v_list, lines)
    endTimeDecoup = time.time()
    print("Final running time for Decoupled method is: ", endTimeDecoup-startTimeDecoup, "seconds")