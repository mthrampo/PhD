## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         Import Libraries and functions                                              #
## ------------------------------------------------------------------------------------------------------------------- #
## =====================================================================================================================

import pandas as pd
import numpy as np

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         assignRetrofitOption function                                               #
## ------------------------------------------------------------------------------------------------------------------- #
##              Description: This function retrieves the categorical classes for the retrofit selction.                #
##                           More specifically it is given as input a samples x 7 binary matrix where                  #
##                           the columns represent the different retrofit option. A 1 in the ith column                #
##                           of the nth row means that the ith retrofit option is selected for the                     #
##                           nth sample building. The output is a samples x 1 matrix and for each sample               #
##                           the appropriate retrofit selection string is assigned.                                    #
##                           Possible retrofit options: 1) No, 2) Full, 3) Wall, 4) Window, 5) Wall & Window           #
##                           6) Roof, 7) Roof & Wall & Winndow                                                         #
## =====================================================================================================================

def assignRetrofitOption(ret_DF):

    #print ('Assigning retrofit options...')

    # Retrieve data frame format
    ret_DF_col = ret_DF.columns
    ret_DF_index = ret_DF.index

    retOption_colName = ['Base', 'Full', 'Wall', 'Window', 'Wall & Window', 'Roof','Roof & Wall & Window']
    retOption = ret_DF_col

    # Change index
    retCol_DF_index = np.arange(1, len(ret_DF_index) + 1)
    retCol_DF = pd.DataFrame(index = retCol_DF_index, columns = ['Ret.Option'])

    #print('...Create retrofit options...')
    temp_DF = ret_DF.copy()
    temp_DF.index = np.arange(1, len(ret_DF_index) + 1)

    count = 1;
    for retOpt,retOpt_colN in zip(retOption,retOption_colName):
        #print (count)
        retOptionOnes = temp_DF.loc[temp_DF[retOpt] == 1]
        retCol_DF.loc[retOptionOnes.index] = retOpt_colN
        count += 1


    #print('...Done...')

    retCol_DF.index = ret_DF_index

    return retCol_DF

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         assignSystemOption function                                                 #
## ------------------------------------------------------------------------------------------------------------------- #
##              Description: This function retrieves the categorical classes for the system selction.                  #
##                           More specifically it is given as input a samples x 12 matrix where                        #
##                           the columns represent the different system selections. In the ith column                  #
##                           of the nth row the capacity of the ith selected system is given represented               #
##                           by its capacity for the nth sample building. The output is a samples x 1 matrix           #
##                           and for each sample the appropriate system selection string is assigned.                  #
##                           Possible system options: 1) Oil boiler, 2) Gas boiler, 3) Bio Boiler, 4) CHP,             #
##                           5) ASHP, 6) GSHP, 7) Gas boiler (existing), 8)Oil boiler (existing),                      #
##                           9) Bio boiler(existing), 10) Distict heating (existing), 11) Electricity (existing),      #
##                           12)Heat pump (existing)
## =====================================================================================================================

def assignSystemOption(sys_DF):

    #print ('Assigning system options...')

    # Retrieve data frame format
    sys_DF_col = sys_DF.columns
    sys_DF_index = sys_DF.index

    sysOption = sys_DF_col
    sysOption_colName = sysOption

    # Change index
    sysCol_DF_index = np.arange(1, len(sys_DF_index) + 1)
    sysCol_DF = pd.DataFrame(index = sysCol_DF_index, columns = ['Sys.Option'])

    #print('...Create system options...')
    temp_DF = sys_DF.copy()
    temp_DF.index = np.arange(1, len(sys_DF_index) + 1)

    count = 1;
    for sysOpt,sysOpt_colN in zip(sysOption,sysOption_colName):
        #print (count)
        sysOptionOnes = temp_DF.loc[temp_DF[sysOpt] > 0]
        sysCol_DF.loc[sysOptionOnes.index] = sysOpt_colN
        count += 1


    #print('...Done...')

    sysCol_DF.index = sys_DF_index

    return sysCol_DF

