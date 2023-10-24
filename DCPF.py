import pandas as pd
import numpy as np
def DCPF(yBus):
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
    def calcPDC_slack(slack_bus, yBus, busnum, xmat):
        slack_bus = int(slack_bus)
        sum = 0
        for j in range(int(busnum)):
            if j != slack_bus:
                PDC_slack= (yBus[slack_bus][j])*(xmat[slack_bus]- xmat[j])
        return PDC_slack

PDC_slack=calcPDC_slack(slack_bus,yBus,busnum,xmat)
return(slack_bus, xmat, PDC_slack)