import pandas as pd
# %%DCPF
#Y_bus from previous code
#yBus

#Remove row and column corresponding to slack bus
##check excel for bus_type
slack = pd.read_excel('ex_nr.xlsx', sheet_name='initial', index_col='bus_type')

##Find row corresponding to slack
row_index = slack.index[slack['bus_type']=='slack'].tolist()[0]

##Find bus corresponding to sclak
slack_bus=slack[slack_index]

#Drop row/col corresponding to slack
droprow=slack.drop([row_index],axis=0)
dropcol=droprow.drop(columns=slack.columns[col_index])

yBus_woslack=dropcol.values

print('yBus_woslack')
#Neglect the real part of the Ybus elements

#Remove j from imaginary parts

#Multiply the matrix by -1

# %%
