def calcDCPF():
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

    #Find slack_bus
    filename = 'C:/Users/Uxue/Desktop/TETE4205/PowerFlow/ex_nr_ex1.xlsx'
    #Remove row and column corresponding to slack bus
    ##check excel for bus_type and get index=slack_bus
    initial= pd.read_excel(filename, sheet_name='initial')
    bus_type=initial.loc[:,'bus_type']
    slack_bus=np.where(bus_type=='slack')[0][0]

    # Obtain the yBus
    yBus = [[VarMat() for i in range(int(busnum))] for j in range(int(busnum))]
    zBus = [[complex(0, 0) for i in range(int(busnum))] for j in range(int(busnum))]
    getZYbus(busnum, yBus, zBus, line_z)
    y_mini = [[complex(0, 0) for i in range(4)] for j in range(lines.size)]
    piLine(knownnum, r_list, x_list, x_shunt, y_mini, lines, t_x, t_a)
    yBusCutsemCalc(busnum, y_mini, lines, yBus)

    #Remove slack bus from Ybus
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
    PDC = [0 for i in range(busnum)]
    for i in range(int(busnum)):
        if j == i:
            PDC[i] = 0
        else:
            PDC[i] += (yBus[i][j])*(xmat[i]- xmat[j])

    return slack_bus, xmat, PDC

def printDCPF():
    #Read the excel and save variables
    stuff = loadFile('ex_nr_ex1.xlsx')
    knownnum = stuff[9]
    busnum = stuff[10]
    knowns = [VarMat() for i in range(int(knownnum))]
    yBus = [[VarMat() for i in range(int(busnum))] for j in range(int(busnum))]

    slack_bus=calcDCPF()
    print('Slack bus:', slack_bus)
    xmat=calcDCPF()
    print('Angles (º):',xmat)
    PDC=calcDCPF()
    print('Active Power (pu):', PDC)