# Functions file for Power FLow 4205 Project
from classes import *
import numpy as np
import pandas as pd


'''
Function: load file
load initial excel file to get needed lists/info for the rest of the code
'''
def loadFile(filename):
    #read excel file
    filename = '/Users/gracedepietro/Desktop/4205/project/PowerFlow/' + filename
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
    # print("Initial things we know: ")
    # print("Number of buses: ", busnum)
    # print("List of Vs: ", v_list)
    # print("List of Ts: ", t_list)
    # print("List of Ps: ", p_list)
    # print("List of Qs: ", q_list)


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
    return v_list, t_list, p_list, q_list, lines, r_list, x_list, r_shunt, x_shunt, knownnum, busnum, line_z, numT, numV, t_x, t_a

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
def setInitGuess(knownNum, xmat):
    for i in range(int(knownNum)):
        if "V" in xmat[i].name:
            xmat[i].val = 1
        elif "T" in xmat[i].name:
            xmat[i].val = 0


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
                #print(mat[i][j].name + ", " + str(mat[i][j].val))
                str_print[i] += mat[i][j].name + ": " + str(mat[i][j].val)
            else:
                #print(mat[i][j].name + ", " + str(mat[i][j].val) + ", " + str(mat[i][j].type))
                str_print[i] += mat[i][j].name + ": " + str(mat[i][j].val)
            if j != int(num):
                str_print[i] += ', '
        print(str_print[i])


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
#fixme: change z_imp to the individual lists
def getZYbus(busnum, yBus, zBus, z_imp):
    countz = 0
    for i in range(int(busnum)):
        for j in range(int(busnum)):
            yBus[i][j].name = "Y" + str(i + 1) + str(j + 1)
            #not needed anymore
            #ask for "zbus" values
            # if j != i:
            #     if zBus[j][i] != complex(0, 0) or zBus[j][i] != 0:
            #         zBus[i][j] = zBus[j][i]
            #     else:
            #         if z_imp.loc[countz, 'line'] == int(str(i + 1) + str(j + 1)):
            #             a = z_imp.loc[countz, 'R']
            #             b = z_imp.loc[countz, 'X']
            #             countz += 1
            #             zBus[i][j] = complex(a, b)


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
Function: pi line model
for creating a 2x2 admittance matrix for a one line system. Part 1 of the Cutsem method.
'''
def piLine(knownnum, r_list, x_list, x_shunt, y_mini, lines, t_x, t_a):
    line_num = lines.size
    #shape: 0 is 11, 1 is 12, 2 is 21, and 3 is 22
    for i in range(line_num):
        if np.isnan(r_list[i]) == False:
            if x_shunt[i] != '' and x_shunt[i] != 0:
                y_mini[i][0] = 1/complex(r_list[i],x_list[i])+(complex(0,x_shunt[i])/2)
            else:
                y_mini[i][0] = 1 / complex(r_list[i], x_list[i])
            y_mini[i][1] = -1/complex(r_list[i],x_list[i])
            y_mini[i][2] = y_mini[i][1]
            y_mini[i][3] = y_mini[i][0]
        else:
            y_mini[i][0] = 1/complex(0, t_x[i])/(t_a[i]**2)
            y_mini[i][1] = -1/complex(0, t_x[i])/t_a[i]
            y_mini[i][2] = y_mini[i][1]
            y_mini[i][3] = 1/complex(0, t_x[i])



'''
Function: Ybus calculations with cutsem algorithm/pi line model
'''
def yBusCutsemCalc(busnum, y_mini, lines, yBus):
    for i in range(busnum):
        for j in range(busnum):
            if i == j: #diagonal element should be sum of all Y11 minis that connect to said i value
                sum = 0
                for k in range(lines.size): #loop through lines (k will also be the index of y_mini matrices)
                    str_temp = str(lines[k]) #line number like '12' or '23'
                    if str_temp[0] == str(i+1): #if in the list of lines, this specific line starts with i 'ik'
                        sum += y_mini[k][0] #add first element (Y11) of the pi-line y_mini
                    elif str_temp[1] == str(i+1): #or if its line 'ki'
                        sum += y_mini[k][0] #add first element (Y11) of the pi-line y_mini
                yBus[i][j].val = sum
            else: #off diagonal element should be the off-diagonal element of y_mini from bus i to j
                #first check if this line exists to see if the mini Y exists
                str_ind1 = int(str(i+1) + str(j+1))
                str_ind2 = int(str(j+1) + str(i+1))
                mini_index1 = np.where(lines == str_ind1)
                mini_index2 = np.where(lines == str_ind2)
                if str_ind1 in lines:
                    yBus[i][j].val = y_mini[mini_index1[0][0]][1] #get Y12 value of the sub bus
                elif str_ind2 in lines:
                    yBus[i][j].val = y_mini[mini_index2[0][0]][1]
                else:
                    yBus[i][j].val = complex(0,0)


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
def calcPVal(num, yBus, busnum, T, V):
    num = int(num)
    p = (V[num]**2) * yBus[num][num].val.real
    sum = 0
    for j in range(int(busnum)):
        if j != num:
            sum += V[j] * tij(yBus[num][j].val.real, yBus[num][j].val.imag, T[num], T[j])
    return p + (-V[num] * sum)


'''
Function: calculate Q value
'''
def calcQVal(num, yBus, busnum, T, V):
    num = int(num)
    q = -(V[num]**2) * yBus[num][num].val.imag
    sum = 0
    for j in range(busnum):
        if j != num:
            sum += V[j] * uij(yBus[num][j].val.real, yBus[num][j].val.imag, T[num], T[j])
    return q + (-V[num] * sum)


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
            temp = uij(yBus[i][j].val.real, yBus[i][j].val.imag, T[i], T[j])
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
Function: iterate
should update P, Q, known matrix, unknown matrix, and jacobian
'''
def iterate(knownnum, jacobian, ybus, t_list, v_list, knowns, xmat, busnum, qnum1, qnum2):
    #first calculate the jacobian matrix
    calcJacElems(knownnum, jacobian, ybus, t_list, v_list, busnum)
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
            new_p = calcPVal(num, ybus, busnum, t_list, v_list)
            net_injections[i] = new_p ###this is somehow changing knowns too
            new_knowns[i] = new_knowns[i] - new_p
        else:
            #Note: change generating/not +/- for P and Q IN EXCEL
            new_q = calcQVal(num, ybus, busnum, t_list, v_list)
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

    q_limit1 = calcQVal(qnum1-1, ybus, busnum, t_list, v_list)
    q_limit2 = calcQVal(qnum2-1, ybus, busnum, t_list, v_list)
    # get other values of P and Q
    # for i in range(busnum):
    #     if np.isnan(p_list[i]):
    #         p_list[i] = calcPVal(i, ybus, busnum, t_list, v_list)
    #     if np.isnan(q_list[i]):
    #         q_list[i] = calcQVal(i, ybus, busnum, t_list, v_list)

    return corrections, new_knowns, q_limit1, q_limit2

def newtonRhapson(conv_crit, qlim_no1, qlim_no2, qlim_val):
    stuff = loadFile('ex_nr_ex1.xlsx')
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
    setInitGuess(knownnum, xmat)


    yBus = [[VarMat() for i in range(int(busnum))] for j in range(int(busnum))]
    zBus = [[complex(0, 0) for i in range(int(busnum))] for j in range(int(busnum))]
    getZYbus(busnum, yBus, zBus, line_z)


    y_mini = [[complex(0, 0) for i in range(4)] for j in range(lines.size)]
    piLine(knownnum, r_list, x_list, x_shunt, y_mini, lines, t_x, t_a)
    yBusCutsemCalc(busnum, y_mini, lines, yBus)
    #printMultiMat(busnum, yBus, False)
    jacobian = [[JacElem() for i in range(int(knownnum))] for j in range(int(knownnum))]
    nameJacElem(knownnum, knowns, xmat, jacobian)
    convergence = False
    itno = 0
    while not (convergence or itno > 10):
        itno += 1
        print("\n\nIteration #" + str(itno))
        temp_knowns = knowns
        outputs = iterate(knownnum, jacobian, yBus, t_list, v_list, temp_knowns, xmat, busnum, qlim_no1, qlim_no2)
        corrections = outputs[0]
        rhs = outputs[1]
        qlim1 = outputs[2]
        qlim2 = outputs[3]
        print("Jacobian: ")
        printMultiMat(knownnum, jacobian, True)
        print("Corrections Vector (dV, dT): ")
        print(corrections)
        print("RHS Vector (dP, dQ): ")
        print(rhs)
        print("New voltage angles and magnitudes")
        printMat(knownnum, xmat)
        # if qlim1 > qlim_val:
        #     #type-switch
            


        count = 0
        for i in range(corrections.size):
            if abs(corrections[i]) > conv_crit:
                cur = abs(corrections[i])
                count += 1
        convergence = count == 0
    for i in range(busnum):
        if np.isnan(p_list[i]):
            p_list[i] = calcPVal(i, yBus, busnum, t_list, v_list)
        if np.isnan(q_list[i]):
            q_list[i] = calcQVal(i, yBus, busnum, t_list, v_list)
    for i in range(busnum):
        print("P", i + 1, ": ", "{:.4f}".format(p_list[i]), "\t", "Q", i + 1, ": ", "{:.4f}".format(q_list[i]))

def DCPF(yBus,busnum, knowns):
    filename = 'C:/Users/Uxue/Desktop/TETE4205/PowerFlow/ex_nr.xlsx'
    #Remove row and column corresponding to slack bus
    ##check excel for bus_type and get index=slack_bus
    initial= pd.read_excel(filename, sheet_name='initial')
    bus_type=initial.loc[:,'bus_type']
    slack_bus=np.where(bus_type=='slack')[0][0]
    y_bus_without_slack = np.delete(np.delete(yBus, slack_bus, axis=0), slack_bus, axis=1)

    #Neglect the real part of the Ybus elements
    for i in range(y_bus_without_slack.shape[0]):
        for j in range(y_bus_without_slack.shape[1]):
            complex_element = y_bus_without_slack[i, j]
            real_part = np.real(complex_element)
            imaginary_part = np.imag(complex_element)
            # Set the real part to zero
            y_bus_without_slack[i, j] = complex(imaginary_part)

    #Delete the symbol j from the imaginary numbers left
    y_bus_without_slack_magnitude=np.abs(y_bus_without_slack)

    #Multiply the matrix by -1
    yBusDC=-1*y_bus_without_slack_magnitude

    #DC Load Flow
    knowns_without_slack = np.delete(knowns, slack_bus, axis=0)#delete slack bus
    inv_yBusDC=np.linalg.inv(yBusDC)
    xmat = np.matmul(inv_yBusDC, knowns_without_slack)
    new_row = np.zeros((1, xmat.shape[1]))
    xmat = np.insert(xmat, slack_bus, new_row, axis=0)

    #calculate P in slack bus
    slack_bus = int(slack_bus)
    sum = 0
    PDC_slack = 0
    for j in range(int(busnum)):
        if j != slack_bus:
            PDC_slack += (yBus[slack_bus][j])*(xmat[slack_bus]- xmat[j])

    return slack_bus, xmat, PDC_slack






'''
Function: loadbustype
gets an string array with the bus types
'''
def loadbustype(filename):
    #read excel file
    filename = 'C:/Users/Ximena/Desktop/project tet4205/PowerFlow/' + filename
    initial = pd.read_excel(filename, sheet_name='initial', index_col='bus_num')
    #extract bus type as string
    type_list = initial.loc[:, 'bus_type'].to_numpy()
    type_list = np.array(type_list)
    type_list = type_list.astype('str')
    return type_list

'''
Function: iterate_FDLF
the angles are updated halfway to then obtain Q mismatches and update the Vs
'''
def iterate_FDLF(knownnum, ybus, bp_inv, bpp_inv, xmat, slackbus, pvbus, t_list, v_list, knowns, busnum):
    new_knowns = [VarMat() for i in range(knownnum)]
    net_injections = [VarMat() for i in range(knownnum)]
    for i in range(knownnum):
        new_knowns[i].name = knowns[i].name
        new_knowns[i].val = knowns[i].val
    # active power mismatches
    dP_V = [0 for i in range(busnum-len(slackbus))]
    corrections = [VarMat() for i in range(knownnum)]
    cue = 0
    for i in range(knownnum):
        num = int(knowns[i].name[1])-1
        type = knowns[i].name[0]
        net_injections[i].name = knowns[i].name
        if type == 'P':
            net_injections[i].val = calcPVal(num, ybus, busnum, t_list, v_list)
            new_knowns[i].val = new_knowns[i].val - net_injections[i].val
            dP_V[cue] = new_knowns[i].val / v_list[num]
            cue =+ 1
            corrections[i].name = 'T' + knowns[i].name[1]
    # solve for dT
    dT = -np.dot(bp_inv, np.transpose(dP_V))
    # update the angles
    cue = 0
    for j in range(knownnum):
        if xmat[j].name[0] == "T":
            corrections[j].val = dT[cue]
            xmat[j].val += corrections[j].val
            t_list[(int(xmat[j].name[1]) - 1)] = xmat[j].val
            cue = + 1

    # reactive power mismatches (using updated values of T)
    dQ_V = [0 for i in range(busnum - len(slackbus) - len(pvbus))]
    cue = 0
    for i in range(knownnum):
        num = int(knowns[i].name[1]) - 1
        type = knowns[i].name[0]
        if type == 'Q':
            net_injections[i].val = calcQVal(num, ybus, busnum, t_list, v_list)
            new_knowns[i].val = new_knowns[i].val - net_injections[i].val
            dQ_V[cue] = new_knowns[i].val / v_list[num]
            cue = + 1
            corrections[i].name = 'V' + knowns[i].name[1]
    # solve for dV
    dV = -np.dot(bpp_inv, np.transpose(dQ_V))
    # update the voltages
    cue = 0
    for j in range(knownnum):
        if xmat[j].name[0] == "V":
            corrections[j].val = dV[cue]
            xmat[j].val += corrections[j].val
            v_list[(int(xmat[j].name[1]) - 1)] = xmat[j].val
            cue = + 1

    return corrections, new_knowns


'''
Function: iterate_FDLF_endit
updates of angles and voltages are done at the end of the iteration, after getting P and Q mismatches
'''
def iterate_FDLF_endit(knownnum, ybus, bp_inv, bpp_inv, xmat, slackbus, pvbus, t_list, v_list, knowns, busnum):
    new_knowns = [VarMat() for i in range(knownnum)]
    net_injections = [VarMat() for i in range(knownnum)]
    for i in range(knownnum):
        new_knowns[i].name = knowns[i].name
        new_knowns[i].val = knowns[i].val
    # active and reactive power mismatches (not updating values of T)
    dP_V = [0 for i in range(busnum-len(slackbus))]
    dQ_V = [0 for i in range(busnum - len(slackbus) - len(pvbus))]
    corrections = [VarMat() for i in range(knownnum)]
    cueP = 0
    cueQ = 0
    for i in range(knownnum):
        num = int(knowns[i].name[1])-1
        type = knowns[i].name[0]
        net_injections[i].name = knowns[i].name
        if type == 'P':
            net_injections[i].val = calcPVal(num, ybus, busnum, t_list, v_list)
            new_knowns[i].val = new_knowns[i].val - net_injections[i].val
            dP_V[cueP] = new_knowns[i].val / v_list[num]
            cueP =+ 1
            corrections[i].name = 'T' + knowns[i].name[1]
        else:
            net_injections[i].val = calcQVal(num, ybus, busnum, t_list, v_list)
            new_knowns[i].val = new_knowns[i].val - net_injections[i].val
            dQ_V[cueQ] = new_knowns[i].val / v_list[num]
            cueQ = + 1
            corrections[i].name = 'V' + knowns[i].name[1]
    # solve for dT and dV
    dT = -np.dot(bp_inv, np.transpose(dP_V))
    dV = -np.dot(bpp_inv, np.transpose(dQ_V))
    # update the angles and voltages
    cueP = 0
    cueQ = 0
    for j in range(knownnum):
        if xmat[j].name[0] == "T":
            corrections[j].val = dT[cueP]
            xmat[j].val += corrections[j].val
            t_list[(int(xmat[j].name[1]) - 1)] = xmat[j].val
            cueP = + 1
        else:
            corrections[j].val = dV[cueQ]
            xmat[j].val += corrections[j].val
            v_list[(int(xmat[j].name[1]) - 1)] = xmat[j].val
            cueQ = + 1
    return corrections, new_knowns

'''
Function: FastDecoupled
algorithm with the Fast Decoupled method
'''
def FastDecoupled(conv_crit):
    # Read the excel and save variables
    stuff = loadFile('ex_nr.xlsx')
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
    knowns = [VarMat() for i in range(int(knownnum))]
    xmat = [VarMat() for j in range(int(knownnum))]
    getInitMats(xmat, knowns, p_list, q_list, busnum)
    setInitGuess(knownnum, xmat)

    # Obtain the yBus
    yBus = [[VarMat() for i in range(int(busnum))] for j in range(int(busnum))]
    zBus = [[complex(0, 0) for i in range(int(busnum))] for j in range(int(busnum))]
    getZYbus(busnum, yBus, zBus, line_z)
    y_mini = [[complex(0, 0) for i in range(4)] for j in range(lines.size)]
    piLine(knownnum, r_list, x_list, x_shunt, y_mini, lines)
    yBusCutsemCalc(busnum, y_mini, lines, yBus)
    print("YBus: ")
    printMultiMat(busnum, yBus, False)

    # Get slack and pv buses
    type_list = loadbustype('ex_nr.xlsx')
    slackbus = np.where(type_list == 'slack')[0]
    pvbus = np.where(type_list == 'pv')[0]

    # Obtain the inverses of B' and B''
    bp = np.empty([busnum-len(slackbus), busnum-len(slackbus)])
    bp[:] = np.nan
    bpp = np.empty([busnum - len(slackbus) - len(pvbus), busnum-len(slackbus)-len(pvbus)])
    bpp[:] = np.nan
    ic = 0
    jc = 0
    icc = 0
    jcc = 0
    for i in range(busnum):
        for j in range(busnum):
            if i not in slackbus and j not in slackbus:
                bp[ic, jc] = yBus[i][j].val.imag
                if jc < busnum - len(slackbus) - 1:
                    jc += 1
                else:
                    jc = 0
                    ic += 1
                if i not in pvbus and j not in pvbus:
                    bpp[icc, jcc] = yBus[i][j].val.imag
                    if jcc < busnum - len(slackbus) - len(pvbus) - 1:
                        jcc += 1
                    else:
                        jcc = 0
                        icc += 1
    bp_inv = np.linalg.inv(bp)
    bpp_inv = np.linalg.inv(bpp)

    # iteration process
    convergence = False
    itno = 0
    while not convergence:
        itno += 1
        print("Iteration #" + str(itno))
        temp_knowns = knowns
        # option of iterate with "iterate_FDLF" or "iterate_FDLF_endit"
        outputs = iterate_FDLF(knownnum, yBus, bp_inv, bpp_inv, xmat, slackbus, pvbus, t_list, v_list, temp_knowns,
                               busnum)
        corrections = outputs[0]
        rhs = outputs[1]
        print("RHS Vector: ")
        printMat(knownnum, rhs)
        print("New voltage angles and magnitudes")
        printMat(knownnum, xmat)
        count = 0
        for i in range(knownnum):
            if abs(corrections[i].val) > conv_crit:
                cur = abs(corrections[i].val)
                count += 1
        convergence = count == 0

    # get other values of P and Q
    for i in range(busnum):
        if np.isnan(p_list[i]):
            p_list[i] = calcPVal(i, yBus, busnum, t_list, v_list)
        if np.isnan(q_list[i]):
            q_list[i] = calcQVal(i, yBus, busnum, t_list, v_list)





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
should update P, Q, known matrix, unknown matrix, and jacobian for DLF
'''
def iterateDLF(knownnum, jacobian, ybus, t_list, v_list, knowns, xmat, busnum):
    #first calculate the jacobian matrix
    calcJacElemsDLF(knownnum, jacobian, ybus, t_list, v_list, busnum)
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
            # subtractions may be wrong
            new_p = calcPVal(num, ybus, busnum, t_list, v_list)
            new_knowns[i] = -new_p - new_knowns[i]
        else:
            #FIXME: same as P
            # subtractions may be wrong
            new_knowns[i] = -calcQVal(num, ybus, busnum, t_list, v_list) - new_knowns[i]
    #now solve for the new values
    #get temp jac of just values oops
    # print("new knowns")
    # for i in range(knownnum):
    #     print(new_knowns[i])
    temp_jac = [[0 for i in range(int(knownnum))] for j in range(int(knownnum))]
    for i in range(knownnum):
        for j in range(knownnum):
            temp_jac[i][j] = jacobian[i][j].val
    corrections = np.linalg.solve(temp_jac, new_knowns)
    # print("corrections")
    # for i in range(knownnum):
    #     print(corrections[i])
    for j in range(knownnum):
        xmat[j].val += corrections[j] #this is wrong
        temp_num = int(xmat[j].name[1])
        temp_val = xmat[j].val
        if xmat[j].name[0] == "T":
            t_list[(temp_num-1)] = temp_val
        elif xmat[j].name[0] == "V":
            v_list[(temp_num-1)] = temp_val
        else:
            print("error thrown in updating v and t lists")
    return corrections
#
# def DecoupledLoadFlow(conv_crit):
#     stuff = open('ex_nr.xlsx','r')
#     convergence = False
#     itno = 0
#     while not convergence:
#         itno += 1
#         print("\n\nIteration #" + str(itno))
#         temp_knowns = knowns
#         outputs = iterate(knownnum, jacobian, yBus, t_list, v_list, temp_knowns, xmat, busnum)
#         corrections = outputs[0]
#         rhs = outputs[1]
#         print("Jacobian: ")
#         printMultiMat(knownnum, jacobian, True)
#         print("Corrections Vector (dV, dT): ")
#         print(corrections)
#         print("RHS Vector (dP, dQ): ")
#         print(rhs)
#         print("New voltage angles and magnitudes")
#         printMat(knownnum, xmat)
#         count = 0
#         for i in range(corrections.size):
#             if abs(corrections[i]) > conv_crit:
#                 cur = abs(corrections[i])
#                 count += 1
#         convergence = count == 0
