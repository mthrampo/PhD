from sklearn.externals import joblib
import pandas as pd


def save_GridSearchCV_object(gs_object, savePath):

    gs_bestEstimator_fileN = 'GS_obj.pkl'
    joblib.dump(gs_object, savePath + gs_bestEstimator_fileN)
    print('File saved successfully!')
    return gs_bestEstimator_fileN


def load_GridSearchCV_object(loadPath, fileName):
    gs_bestEstimator_fileN = fileName
    gs_object = joblib.load(loadPath + gs_bestEstimator_fileN)
    print('File loaded successfully!')
    return gs_object

def save_DataFrame_object(dataFrame_object, keyWord ,savePath):
    gs_bestEstimator_predictions_fileN = 'dataFrame_'+keyWord+'.pkl'
    dataFrame_object.to_pickle(savePath + gs_bestEstimator_predictions_fileN)
    print('File saved successfully!')
    return gs_bestEstimator_predictions_fileN

def load_DataFrame_object(loadPath, fileName):
    gs_bestEstimator_predictions_fileN = fileName
    dataFrame_object = pd.read_pickle(loadPath + gs_bestEstimator_predictions_fileN)
    print('File loaded successfully!')
    return dataFrame_object

