## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         Import Libraries and functions                                              #
## ------------------------------------------------------------------------------------------------------------------- #
## =====================================================================================================================

import pandas as pd
import numpy as np
import math as mt
import itertools as it
import operator as op

from constructionAge import assignConstructionAgeClass

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from keras.utils import to_categorical

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                        Continuous feature normalization                                             #
## ------------------------------------------------------------------------------------------------------------------- #
##          Description: This function normalizes the continuous features. It supports two methods:                    #
##                       1) Normalize between 0 and 1                                                                  #
##                       2) Replace data to look like standard normally distributed (zero mean & unit variance)        #
## =====================================================================================================================


def continuousNormalization(continuousData,norm_method):

    contData_DF_col =  continuousData.columns
    contData_DF_index = continuousData.index

    if norm_method == 'minMax':
        contScaler = MinMaxScaler()
    elif norm_method == 'stand':
        contScaler = StandardScaler()

    contScaler_info = contScaler.fit(continuousData)
    norm_contData = contScaler_info.transform(continuousData)
    norm_contData_DF = pd.DataFrame(norm_contData, columns = contData_DF_col,index = contData_DF_index)

    return (norm_contData_DF, contScaler_info)

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                        Categorical feature encoding                                                 #
## ------------------------------------------------------------------------------------------------------------------- #
##          Description: This function performs feature encoding to categorical data. Special treatment                #
##                       is given to data including construction age information. The existance of such information    #
##                       is defined by the ageIncluded boolean. The input data and the selecte encoding method         #
##                       are also passed as inputs. It supports the following methods:                                 #
##                       1) Label encoding: Each class is replaced with an integer number.                             #
##                       2) One-hot encoding: Use of d bits to represent each class, where d is the                    #
##                                            number of classes. The ith bit is one for the ith category               #
##                                            and all other bits are set to zero.                                      #
##                       3) Binary encoding: Binary representation of the classes with the least possible              #
##                                           number of necessary bits. Calls custom function                           #
## =====================================================================================================================

def categoricalEncoding(inputCatData, enc_method, ageIncluded):

    catData = inputCatData.copy()

    if ageIncluded == 'Yes':                                                                                            # Assign construction age classes (as given from arcGIS data)
        exactBuildingAge = catData[['Age']]
        buildingConstrAgeClass = assignConstructionAgeClass(exactBuildingAge)
        buildingConstrAgeClass_DF = pd.DataFrame(buildingConstrAgeClass, index= catData.index, columns=['Age'])
        catData.loc[:,['Age']] = buildingConstrAgeClass_DF.values

    catData_DF_col = catData.columns
    catData_DF_index = catData.index

    enc_catData_DF = pd.DataFrame()
    catData_enc_info = {}

    if enc_method == 'label':                                                                                           # Label encoding
        for col in catData.columns:
            (temp_DF,catEncoder_info) = labelEncoding(catData[col],catData_DF_index)                                    #   - Call labelEncoding custom function
            enc_catData_DF = pd.concat([enc_catData_DF, temp_DF], axis=1)
            catData_enc_info[col + 'Enc.Info'] = catEncoder_info
        enc_catData_DF.columns = catData_DF_col

        return enc_catData_DF, catData_enc_info                                                                         #   - Return encoded data and encoding info

    elif enc_method == 'one-hot':                                                                                       # One-hot encoding
        rep_input = input("  -- What kind of representation? [sparse / dense]: ")
        if rep_input == 'sparse':
            sparseInput = True
        elif rep_input == 'dense':
            sparseInput = False
        catEncoder = OneHotEncoder(sparse = sparseInput)

        for col in catData.columns:
            if isinstance(catData[col].iloc[0], str):
                (catData_temp, labelEncoder_info) = labelEncoding(catData[col],catData_DF_index)
                catData_enc_info[col + ' Enc.Info'] = labelEncoder_info
            else:
                catData_temp = catData[col]
            catVar = np.array(catData_temp)
            catEncoder_info = catEncoder.fit(catVar.reshape(-1,1))
            if catData_enc_info.get(col + ' Enc.Info', 0) == 0:
                catData_enc_info[col + ' Enc.Info'] = catEncoder_info
            else:
                catData_enc_info[col + ' Enc.Info'] = (catData_enc_info.get(col + ' Enc.Info', 0), catEncoder_info)
            enc_catData = catEncoder_info.transform(catVar.reshape(-1,1))
        #     enc_catData = to_categorical(catVar)                                                                        # Added for keras one hot encoding
            temp_DF = pd.DataFrame(enc_catData, columns =  [col+' bit_' + str(i) for i in np.arange(1, enc_catData.shape[1] + 1)], index=catData_DF_index)
            enc_catData_DF = pd.concat([enc_catData_DF, temp_DF],axis = 1)

        return enc_catData_DF, catData_enc_info

    elif enc_method == 'binary':
        catData.index = np.arange(1, catData.shape[0] + 1)
        catData_enc_info_DF = pd.DataFrame()
        for col in catData.columns:
            print (col)
            if isinstance(catData[col].iloc[0], str):
                (catData_temp, labelEncoder_info) = labelEncoding(catData[col],catData_DF_index)
                catData_enc_info[col + ' Enc.Info'] = labelEncoder_info
            else:
                catData_temp = catData[col]
            (temp_DF, temp_enc_info) = binaryEncoding(catData_temp, catData_DF_index, col)
            enc_catData_DF = pd.concat([enc_catData_DF, temp_DF], axis=1)
            catData_enc_info_DF = catData_enc_info_DF.combine_first(temp_enc_info)
        return enc_catData_DF, catData_enc_info_DF

    # return enc_catData_DF


def labelEncoding(catVariable, index):
    catEncoder = LabelEncoder()
    catEncoder_info = catEncoder.fit(catVariable)
    enc_catData = catEncoder_info.transform(catVariable)
    enc_catData_DF = pd.DataFrame(enc_catData,index = index) #, columns=catVariable.columns, index = catVariable.index)

    return (enc_catData_DF, catEncoder_info)

def binaryEncoding(catVar, catIndex, catVar_name):

    # Make sure input is pandas series
    if not isinstance(catVar,pd.Series):
        catVar = pd.Series(catVar[0])

    # Get categories that appear in dataset
    valueCounts = catVar.value_counts()
    categories = ((valueCounts.index[0:].sort_values()).astype(np.int64)).get_values()

    # Convert to numpy array
    catVariable = np.array(catVar)

    # Transform categories to start from category 0 (or -1)
    if categories[0]!=0:
        temp_categories = categories - 1*categories[0]
        numOfCategories = max(catVariable)
    else:
        temp_categories = categories
        numOfCategories = temp_categories.shape[0]

    bitSize = mt.ceil(mt.log(numOfCategories) / mt.log(2))
    enc_data = np.zeros([catVariable.shape[0], bitSize])
    binCombinations = [list(i) for i in it.product([0, 1], repeat = bitSize)]
    for index, observedCategory in enumerate(catVariable):
        # print(index)
        for category in temp_categories: #range(int(min(catVariable)),int(numOfCategories)): #+1):
            if observedCategory == category:
                enc_data[index,:] = binCombinations[int(category)] #-1]
                break;
    encData_DF = pd.DataFrame(enc_data, columns = [catVar_name+' bit_' + str(i) for i in np.arange(1, enc_data.shape[1] + 1)], index= catIndex)
    # biComb_strArray = []
    # for binPer in binCombinations:
    #     biComb_strArray.append(''.join(str(x) for x in binPer))

    # Keep bit transformation for all possible categories
    # enc_data_transformationInfo_DF = pd.DataFrame(binCombinations[0:int(numOfCategories)+1-int(min(catVariable))], columns = [catVar.name+' bit_' + str(i) for i in np.arange(1, enc_data.shape[1] + 1)],
    #                                               index = np.arange(int(min(catVariable)),int(numOfCategories)+1))

    # Keep bit transformation only for observed categories
    enc_data_transformationInfo_DF = pd.DataFrame(list(op.itemgetter(*temp_categories)(binCombinations)), columns = [catVar_name+' bit_' + str(i) for i in np.arange(1, enc_data.shape[1] + 1)],
                                                  index = categories)

    return encData_DF, enc_data_transformationInfo_DF











        # boolVector = (catVariable == category)
        # indexVector = [i for i, x in enumerate(boolVector) if x]
        # enc_data[indexVector,:] = binCombinations[int(category)]
        #
        #


    # for count, i in enumerate(catVar):
    #     enc_data[count, :] = list(np.binary_repr(int(i), width=bitSize))