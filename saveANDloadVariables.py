## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         Import Libraries and functions                                              #
## ------------------------------------------------------------------------------------------------------------------- #
## =====================================================================================================================

from sklearn.externals import joblib
import pandas as pd

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         save_GridSearchCV_object function                                           #
## ------------------------------------------------------------------------------------------------------------------- #
##              Description: This function saves in a file the grid search object of a cross validation                #
##                           grid search process on a machine learning algorithm. This objects includes                #
##                           all the information stored while searching for the appropriate parameters.                #
## =====================================================================================================================

def save_GridSearchCV_object(gs_object, savePath):

    gs_bestEstimator_fileN = 'GS_obj.pkl'
    joblib.dump(gs_object, savePath + gs_bestEstimator_fileN)
    print('File saved successfully!')
    return gs_bestEstimator_fileN

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         load_GridSearchCV_object function                                           #
## ------------------------------------------------------------------------------------------------------------------- #
##              Description: This function loads a file that contains a grid search object of a cross validation       #
##                           grid search process on a machine learning algorithm. This objects includes                #
##                           all the information stored while searching for the appropriate parameters.                #
## =====================================================================================================================

def load_GridSearchCV_object(loadPath, fileName):
    gs_bestEstimator_fileN = fileName
    gs_object = joblib.load(loadPath + gs_bestEstimator_fileN)
    print('File loaded successfully!')
    return gs_object

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         save_DataFrame_object function                                              #
## ------------------------------------------------------------------------------------------------------------------- #
##              Description: This function saves in a file dataframe objects with a specific name                      #
##                           denoted by a keyWord that is specified by the user. Mainly used for saving                #
##                           target and output results from the machine learning algorithms for later plotting.        #
## =====================================================================================================================

def save_DataFrame_object(dataFrame_object, keyWord ,savePath):
    gs_bestEstimator_predictions_fileN = 'dataFrame_'+keyWord+'.pkl'
    dataFrame_object.to_pickle(savePath + gs_bestEstimator_predictions_fileN)
    print('File saved successfully!')
    return gs_bestEstimator_predictions_fileN

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         load_DataFrame_object function                                              #
## ------------------------------------------------------------------------------------------------------------------- #
##              Description: This function loads a dataframe. Mainly used for loading target and output results        #
##                           for plotting purposes.                                                                     #
## =====================================================================================================================

def load_DataFrame_object(loadPath, fileName):
    gs_bestEstimator_predictions_fileN = fileName
    dataFrame_object = pd.read_pickle(loadPath + gs_bestEstimator_predictions_fileN)
    print('File loaded successfully!')
    return dataFrame_object

