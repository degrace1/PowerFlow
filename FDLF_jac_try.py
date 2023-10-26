from functions import *
def nameJacElem_FDLF(knownnum, knowns, xmat, jacobian):
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


def calcJacElems_FDLF(knownnum, jacobian, ybus, t_list, v_list, busnum):
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



def iterate_FDLF_jac_end(knownnum, jacobian, ybus, t_list, v_list, knowns, xmat, busnum, qnum1, qnum2):
    #first calculate the jacobian matrix
    calcJacElems_FDLF(knownnum, jacobian, ybus, t_list, v_list, busnum)
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

def iterate_FDLF_jac_mid(knownnum, jacobian, ybus, t_list, v_list, knowns, xmat, busnum, qnum1, qnum2):
    #first calculate the jacobian matrix
    calcJacElems_FDLF(knownnum, jacobian, ybus, t_list, v_list, busnum)

    j1_len = 0
    for j in range(knownnum):
        if jacobian[0][j].type == 'dpidti' or jacobian[0][j].type == 'dpidtj':
            j1_len += 1
    j4_len = int(knownnum-j1_len)
    jac_1 = [[0 for i in range(int(j1_len))] for j in range(int(j1_len))]
    for i in range(j1_len):
        for j in range(j1_len):
            jac_1[i][j] = jacobian[i][j].val
    jac_4 = [[0 for i in range(int(j4_len))] for j in range(int(j4_len))]
    for i in range(j4_len):
        for j in range(j4_len):
            jac_4[i][j] = jacobian[i][j].val

    #make temp knowns matrix without the names
    dP = [0 for i in range(int(j1_len))]
    new_knowns = [0 for i in range(knownnum)]
    net_injections = [0 for i in range(knownnum)]
    corrections = [0 for i in range(int(knownnum))]
    cue = 0
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
            dP[cue] = new_knowns[i]
            cue += 1
    correctionsP = np.linalg.solve(jac_1, dP)
    cue = 0
    for j in range(knownnum):
        if xmat[j].name[0] == "T":
            corrections[j] = correctionsP[cue]
            xmat[j].val += correctionsP[cue]
            temp_num = int(xmat[j].name[1])
            temp_val = xmat[j].val
            t_list[(temp_num - 1)] = temp_val
            cue += 1

    dQ = [0 for i in range(int(j4_len))]
    cue = 0
    for i in range(knownnum):
        #for each known value, calculate the new value of P or Q and subtract them
        num = int(knowns[i].name[1])-1
        type = knowns[i].name[0]
        if type == 'Q':
            #Note: change generating/not +/- for P and Q IN EXCEL
            new_q = calcQVal(num, ybus, busnum, t_list, v_list)
            net_injections[i] = new_q
            new_knowns[i] = new_knowns[i] - new_q
            dQ[cue] = new_knowns[i]
            cue += 1
    correctionsQ = np.linalg.solve(jac_4, dQ)
    cue = 0
    for j in range(knownnum):
        if xmat[j].name[0] == "V":
            corrections[j] = correctionsQ[cue]
            xmat[j].val += correctionsQ[cue]
            temp_num = int(xmat[j].name[1])
            temp_val = xmat[j].val
            v_list[(temp_num-1)] = temp_val
            cue += 1

    q_limit1 = calcQVal(qnum1-1, ybus, busnum, t_list, v_list)
    q_limit2 = calcQVal(qnum2-1, ybus, busnum, t_list, v_list)
    # get other values of P and Q
    # for i in range(busnum):
    #     if np.isnan(p_list[i]):
    #         p_list[i] = calcPVal(i, ybus, busnum, t_list, v_list)
    #     if np.isnan(q_list[i]):
    #         q_list[i] = calcQVal(i, ybus, busnum, t_list, v_list)

    return corrections, new_knowns, q_limit1, q_limit2




def FDLF_jac(conv_crit, qlim_no1, qlim_no2, qlim_val):
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
    nameJacElem_FDLF(knownnum, knowns, xmat, jacobian)
    convergence = False
    itno = 0
    while not (convergence or itno > 10):
        itno += 1
        print("\n\nIteration #" + str(itno))
        temp_knowns = knowns
        outputs = iterate_FDLF_jac_mid(knownnum, jacobian, yBus, t_list, v_list, temp_knowns, xmat, busnum, qlim_no1, qlim_no2)
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
        for i in range(knownnum):
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