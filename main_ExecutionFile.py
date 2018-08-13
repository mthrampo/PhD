## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         Import Libraries and functions                                              #
## ------------------------------------------------------------------------------------------------------------------- #
## =====================================================================================================================

import pandas as pd                                                                                                     # Libraries
import numpy as np

import matplotlib.pyplot as plt
import random

from loadBuildingInfo_matFiles import loadBuildingInfo                                                                  # Functions
from variablesStandardization import continuousNormalization, categoricalEncoding
from assignCategoricalVariables import assignRetrofitOption, assignSystemOption
from ML_algorithms import MLPerceptron, hybridMLAlgorithm, kerasNN
from saveANDloadVariables import save_GridSearchCV_object, load_GridSearchCV_object,save_DataFrame_object,\
    load_DataFrame_object
from results import plotGridSearchComp

print('        ---> STATUS <---')

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                              Import building information (saved in .mat files)                                      #
## ------------------------------------------------------------------------------------------------------------------- #
## =====================================================================================================================

print("--** Import building information **--")

matPath = 'C:/Users/the/Documents/Empa/Empa/Work/NeuralNetworks/NN_Approach_1/'                                         # Path were the .mat files are stored

[buildinfInfo_DF, ED_annual_h_DF, ED_annual_e_DF, ED_peak_h_DF, ED_peak_e_DF, EH_output_DF] = loadBuildingInfo(matPath)

print('...Successful')

# # ## =====================================================================================================================
# # ## ------------------------------------------------------------------------------------------------------------------- #
# # ##                                              Outlier detection                                                      #
# # ## ------------------------------------------------------------------------------------------------------------------- #
# # ## =====================================================================================================================

print("--** Outlier detection **--")

import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# # A1BS_input = pd.concat([input_catData_enc_DF,buildingInfo_norm_DF[['Ground floor area','Height']]], axis = 1)
# #
# # clf = IsolationForest(max_samples=100, random_state=4)
# # clf.fit(A1BS_input)
# # y_pred_train = clf.predict(A1BS_input)
#
#
#
#
# input_df = pd.concat([buildingInfo_contData_DF,buildinfInfo_catData_DF],axis = 1)
#
# # plt.hist(input_df['Age'])
# for col in input_df.columns:
#     bp = input_df.boxplot(fontsize=15, grid=True, figsize=(20, 10), return_type='dict', column=col, showmeans=True)
#     whiskers_values = [item.get_ydata()[1] for item in bp['whiskers']]
#     data = input_df[col]
#     outliers = (data > whiskers_values[1]) # and (data < whiskers_values[1])]
#     for i, bool in enumerate(outliers):
#         if bool == False:
#             input_df.at[i + 1, col] = 0
#
#
#
#
# a = input_df.boxplot(fontsize = 15, grid=True, figsize=(20, 10), return_type = 'dict', column = 'Age', showmeans = True)
# plt.plot()
#
# whiskers_values = [item.get_ydata()[1] for item in a['whiskers']]
# age = input_df['Age']
# age_outliers = (age > whiskers_values[0])
# age_wo_outliers = age[age_outliers]
#
# for i,building in enumerate(age_outliers):
#     if building == False:
#         input_df.at[i+1,'Age'] = 0
#
#
#
#
# print (input_df.describe())
#
# input_df['Age'].quantile(0.3) - 1.5*(input_df['Age'].quantile(0.3)-input_df['Age'].quantile(0.1))

input_db = buildinfInfo_DF[['Age', 'Total Area', 'Heating en.Carrier', 'DHW en.Carrier',
                                         'Ground floor area', 'Height', 'Roof area', 'Roof orientation',
                                         'Roof slope', 'Building type']]

clf = IsolationForest(max_samples=100, random_state=4)                                                                  # Remove outliers
clf.fit(input_db)
outlier_table = np.reshape(clf.predict(input_db),(-1,1)) #,index = A1BS_input.index)
input_db_clear = input_db
# A1BS_outlier_Table = pd.DataFrame()
# for i, outlier in enumerate(outliers.iterrows()):
input_db_outliers = pd.DataFrame(columns = input_db_clear.columns) #np.zeros((1,A1BS_input_clear.shape[0]))
input_db_outliers_id = []

for i, (outlier, info) in enumerate(zip(outlier_table,input_db_clear.iterrows())):
    if outlier == -1:
        # print ('In')
        input_db_outliers.loc[len(input_db_outliers)] = info[1]
        input_db_clear.drop(i+1, inplace=True)
        input_db_outliers_id.append(i+1)

input_db_outliers.index = input_db_outliers_id

print('...outliers removed')

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                            Standardization of variables                                             #
## ------------------------------------------------------------------------------------------------------------------- #
## =====================================================================================================================

print("--** Variable standardization **--")
                                                                                                                        # Continuous variables
energyProfiles_DF = pd.concat([ED_annual_h_DF, ED_annual_e_DF, ED_peak_h_DF, ED_peak_e_DF], axis = 1)                   #   - Concatenate energy profiles
buildingInfo_contData_DF = buildinfInfo_DF[['Ground floor area','Height','Roof area','Roof orientation','Roof slope']]
EH_output_contData_DF = EH_output_DF[['Cost','Emissions','Oil boiler','Gas boiler','Bio boiler','CHP','ASHP','GSHP',
                                      'PV','Solar thermal','Heat storage','Electricity storage','Gas boiler (existing)',
                                      'Oil boiler (existing)','Bio boiler (existing)','Distict heating (existing)',
                                      'Electricity (existing)','Heat pump (existing)']]

# contData_DF = pd.concat([buildingInfo_contData_DF, energyProfiles_DF], axis = 1)

print("--* Continuous Data Normalization *--")

norm_method = input("  - Select normalization method for continuous features [minMax / stand]: ")                       # Ask user to define normalization method
                                                                                                                        # minMax: Rescale between [0,1]
                                                                                                                        # stand: Replace data to look like standard normally
                                                                                                                        #        distributed (zero mean & unit variance)

(buildingInfo_norm_DF, buildingInfo_norm_info) = continuousNormalization(buildingInfo_contData_DF,norm_method)          #   - Normalize building information continuous variables
(profiles_norm_DF, profiles_norm_info) = continuousNormalization(energyProfiles_DF,norm_method)                         #   - Normalize energy profiles
(EH_output_objectiveData_norm_DF, EH_output_objectiveData_info) = \
    continuousNormalization(EH_output_contData_DF[['Cost','Emissions']],norm_method)                                    #   - Normalize energy hub objectives output continuous data
(EH_output_SysCapData_norm_DF, EH_output_SysCapData_info) = \
    continuousNormalization(EH_output_contData_DF[['Oil boiler','Gas boiler','Bio boiler','CHP','ASHP','GSHP',
                                      'PV','Solar thermal','Heat storage','Electricity storage','Gas boiler (existing)',
                                      'Oil boiler (existing)','Bio boiler (existing)','Distict heating (existing)',
                                      'Electricity (existing)','Heat pump (existing)']],norm_method)                    #   - Normalize energy hub system capacities output continuous data
(age_norm_DF, age_norm_info) = continuousNormalization(buildinfInfo_DF[['Age']],norm_method)                            #   - Normalize age (to be used when accounted as continuous)
                                                                                                                        # Categorical variables
print("--* Categorical Data Encoding *--")

enc_method = input("  - Select encoding method for categorical features [label / one-hot / binary]: ")                  # Ask user to define normalization method

                                                                                                                        #   - Construct retrofit and system categorical variables
EH_output_catData_DF = EH_output_DF[['Base case','Full retrofit','Wall retrofit','Window retrofit',
                                     'Wall & Window retrofit','Roof retrofit','Roof & Wall & Window retrofit']]

ret_DF = assignRetrofitOption(EH_output_catData_DF)

system_cap_DF = EH_output_DF[['Oil boiler','Gas boiler','Bio boiler','CHP','ASHP','GSHP','Gas boiler (existing)',
                              'Oil boiler (existing)','Bio boiler (existing)','Distict heating (existing)',
                              'Electricity (existing)','Heat pump (existing)']]

sys_DF = assignSystemOption(system_cap_DF)

retSys_DF = pd.concat([ret_DF, sys_DF],axis = 1)                                                                        #   -- Concatenate retrofit and system selection in
                                                                                                                        #      one variable

buildingInfo_catData_DF = buildinfInfo_DF[['Age','Heating en.Carrier','DHW en.Carrier','Building type']]

(input_catData_enc_DF,input_catData_enc_info) = categoricalEncoding(buildingInfo_catData_DF,enc_method,'Yes')           # Encoding of building info categorical data
# input_catData_enc_DF = categoricalEncoding(buildingInfo_catData_DF,enc_method,'Yes')
(output_catData_enc_DF,output_catData_enc_info) = categoricalEncoding(retSys_DF,enc_method,'No')                        # Encoding of retrofit and system selection
# output_catData_enc_DF = categoricalEncoding(retSys_DF,enc_method,'No')


print('... Successful')

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                         Define test set (unseen samples to measure generalization error)                            #
## ------------------------------------------------------------------------------------------------------------------- #
## =====================================================================================================================

buildingInfo_norm_DF.drop(input_db_outliers_id, inplace=True)
profiles_norm_DF.drop(input_db_outliers_id, inplace=True)                                                               # Drop outliers from inputs
age_norm_DF.drop(input_db_outliers_id, inplace=True)
input_catData_enc_DF.drop(input_db_outliers_id, inplace=True)

input_db_outliers_id_pf = (np.array([[(i-1)*10+j for j in np.arange(1,11)] for i in input_db_outliers_id])).flatten()   # Extend outliers list (*10 pareto points)

paretoIndex = EH_output_objectiveData_norm_DF.index
paretoIndex_new = list(paretoIndex)

EH_output_objectiveData_norm_DF.index = np.arange(1,EH_output_objectiveData_norm_DF.shape[0]+1)                         # Change index from 1,2,..9,10,1,2,...,9,10,1 .. to 1,2,..,400,401,..
EH_output_SysCapData_norm_DF.index = np.arange(1,EH_output_SysCapData_norm_DF.shape[0]+1)
output_catData_enc_DF.index = np.arange(1,output_catData_enc_DF.shape[0]+1)

EH_output_objectiveData_norm_DF.drop(input_db_outliers_id_pf, inplace=True)                                             # Drop outliers from outputs
EH_output_SysCapData_norm_DF.drop(input_db_outliers_id_pf, inplace=True)
output_catData_enc_DF.drop(input_db_outliers_id_pf, inplace=True)


del paretoIndex_new[0:len(input_db_outliers_id_pf)]                                                                     # Restore indices
EH_output_objectiveData_norm_DF.index = paretoIndex_new
EH_output_SysCapData_norm_DF.index = paretoIndex_new
output_catData_enc_DF.index = paretoIndex_new

# indices_remove_list = np.arange(0,len(input_db_outliers_id_pf))
# paretoIndex_new = list(paretoIndex).pop(i for i in indices_remove_list)                                                 # Restore index

print("--** Define test set **--")

samples_size = buildingInfo_norm_DF.shape[0];                                                                                                   # Sample size
testSet_size = np.round(samples_size*0.15)

np.random.seed(4)                                                                                                       # Fix random generator (for reproducability)

testSet_ids = np.sort(np.random.choice(samples_size, int(testSet_size), replace=False))                                 # Generate test set
trainSet_ids = np.setdiff1d(np.arange(samples_size),testSet_ids)                                                        # Genetate training set

print('... Successful')


# # ## =====================================================================================================================
# # ## ------------------------------------------------------------------------------------------------------------------- #
# # ##                                  Approach 1 - Building Simulation ML algorithm                                      #
# # ## ------------------------------------------------------------------------------------------------------------------- #
# # ## =====================================================================================================================
#
# print("--* Approach 1 - Building simulation training *--")
# #
A1BS_input = pd.concat([input_catData_enc_DF,buildingInfo_norm_DF[['Ground floor area','Height']]], axis = 1)           # Input = [ Building age; heating energy carrier;
#                                                                                                                         #           DHW energy carrier; building type;
#                                                                                                                         #           floor area; Building height]
#
#
# clf = IsolationForest(max_samples=100, random_state=4)                                                                  # Remove outliers
# clf.fit(A1BS_input)
# outliers = np.reshape(clf.predict(A1BS_input),(-1,1)) #,index = A1BS_input.index)
# A1BS_input_clear = A1BS_input
# # for i, outlier in enumerate(outliers.iterrows()):
# for i, outlier in enumerate(outliers):
#     if outlier == -1:
#         print ('In')
#         A1BS_input_clear.drop(i+1, inplace=True)
#
#
# # A1BS_input = buildingInfo_norm_DF[['Ground floor area','Height']]                                                       # Input = [ floor area; Building height]
#
# # A1BS_input = pd.concat((age_norm_DF, buildingInfo_norm_DF[['Ground floor area','Height']]), axis = 1)                   # Input = [ age, floor area; Building height]

A1BS_input_trainSet = A1BS_input.iloc[trainSet_ids,:]                                                                   #   - Separate input to training
A1BS_input_testSet = A1BS_input.iloc[testSet_ids,:]                                                                     #     and test set

A1BS_target = profiles_norm_DF[energyProfiles_DF.columns]                                                               # Target = [ Annual heating demand; Annual electricity demand;
                                                                                                                        #  Peak heating demand; Peak electricity demand]



A1BS_target_norm_trainSet = A1BS_target.iloc[trainSet_ids,:]                                                                   #   - Separate target to training
A1BS_target_norm_testSet = A1BS_target.iloc[testSet_ids,:]                                                              #     and test set
#
checkPointPath = 'C:/Users/the/Documents/Empa/Empa/PhD/Results/Approach1_BS_NN_21052018/keras/run_13062018/'
#
random_gen_seed = 4

# (keras_gs_4, gs_history) = kerasNN(A1BS_input_trainSet,A1BS_target_norm_trainSet, True, random_gen_seed, checkPointPath)
# # kerasNN(A1BS_input_trainSet,A1BS_target_norm_trainSet, True, random_gen_seed)
# #
# plt.figure(figsize=(20, 10))
# plt.plot(np.arange(1,51,2),abs(keras_gs_4.cv_results_['mean_train_score']))
# plt.plot(np.arange(1,51,2),abs(keras_gs_4.cv_results_['mean_test_score']))
# plt.title('model performance (one-hot encoding)')
# plt.ylabel('mse', fontsize=18)
# plt.xlabel('hidden neurons', fontsize=18)
# plt.legend(['train', 'validation'], loc= 'center right')#'upper right')
# font = {'family': 'normal',
#         'weight': 'normal',
#         'size': 18}
# plt.rc('font', **font)
# plt.grid()
# plt.show()
# #
# plt.figure(figsize=(20, 10))
# plt.plot(keras_gs_4.best_estimator_.model.model.history.history['mean_squared_error'])                                                 # summarize history for accuracy
# plt.plot(keras_gs_4.best_estimator_.model.model.history.history['val_mean_squared_error'])
# plt.title('model performance')
# plt.ylabel('mse', fontsize=18)
# plt.xlabel('epoch', fontsize=18)
# plt.legend(['train', 'validation'], loc='upper right')
# font = {'family': 'normal',
#         'weight': 'normal',
#         'size': 18}
# plt.rc('font', **font)
# plt.grid()
# plt.show()
#
# plt.plot(keras_gs_4.best_estimator_.model.model.history.history['loss'])                                                # summarize history for loss
# plt.plot(keras_gs_4.best_estimator_.model.model.history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss', fontsize=18)
# plt.xlabel('epoch', fontsize=18)
# plt.legend(['train', 'validation'], loc='upper right')
# font = {'family': 'normal',
#         'weight': 'normal',
#         'size': 18}
# plt.rc('font', **font)
# plt.grid()
# plt.show()
#
#
#
# # (gs) = MLPerceptron(A1BS_input_trainSet,A1BS_target_norm_trainSet,'Regression','True','mse,r2')                         # Training
#
#                                                                                                                         # Path to save trained model (sklearn)
# finalDir_sklearn = ['MinMax_Label_10_log_lbfgs_5To60_adap_0.01/',                                                       #   - All inputs + Label encoding + varying hidden nodes
#             'MinMax_OneHot_10_log_lbfgs_5To60_adap_0.01/',                                                              #   - All inputs + One hot encoding + varying hidden nodes
#             'MinMax_Binary_10_log_lbfgs_5To60_adap_0.01/',                                                              #   - All inputs + Binary encoding + varying hidden nodes
#             'MinMax_10_log_lbfgs_5to60_adap_0.01_NO_CATEGORICAL/',                                                      #   - Continuous inputs + no encoding + varying hidden nodes
#             'MinMax_10_log_lbfgs_5to60_adap_0.01_NO_CATEGORICAL_AgeContinuous/']                                        #   - Continuous inputs (incl. age) + no encoding
#                                                                                                                         #                          + varying hidden nodes
#
#                                                                                                                         # Path to save trained model (keras)
# finalDir_keras = ['1.1/',                                                                                                #   - All inputs + Label encoding + varying hidden nodes
#                   '2.1/',                                                                                                #   - All inputs + One hot encoding + varying hidden nodes
#                   '3.1/']                                                                                                #   - All inputs + Binary encoding + varying hidden nodes
#
# BS_models_save_Path = 'C:/Users/the/Documents/Empa/Empa/PhD/Results/Approach1_BS_NN_21052018/keras/'+finalDir_keras[0]
#
# fileName = save_GridSearchCV_object(gs,BS_models_save_Path)                                                             # Save training grid search results
#
# # # serialize model to YAML                                                                                             # Save approach for keras
# # model_yaml = keras_gs_3.to_yaml()
# # with open(BS_models_save_Path+'1_1.yaml', 'w') as yaml_file:
# #     yaml_file.write(model_yaml)
# # # serialize weights to HDF5
# # keras_gs_3.save_weights(BS_models_save_Path+'1_1.h5')
# # print("Saved model to disk")
#
# import pickle
# with open(BS_models_save_Path+'GS_obj.pkl', 'wb') as f:
#     pickle.dump(keras_gs_3, f) # save the object to a file
#
#
#
# A1BS_norm_output_testSet = keras_gs_3.best_estimator_.predict(A1BS_input_testSet)                                               # Prediction (on test set) with best estimator
# A1BS_norm_output_testSet_DF = pd.DataFrame(A1BS_norm_output_testSet, index = A1BS_target_norm_testSet.index,            #   - DF format
#                                            columns = A1BS_target_norm_testSet.columns)
#
# A1BS_output_testSet = profiles_norm_info.inverse_transform(A1BS_norm_output_testSet_DF)                                 # Inverse normalization process for output data
# A1BS_output_testSet_DF = pd.DataFrame(A1BS_output_testSet,index = A1BS_target_norm_testSet.index,                       #   - DF format
#                                       columns = A1BS_target_norm_testSet.columns)
#
# A1BS_target_testSet = profiles_norm_info.inverse_transform(A1BS_target_norm_testSet)                                    # Inverse normalization process for respective target data
# A1BS_target_testSet_DF = pd.DataFrame(A1BS_target_testSet,index = A1BS_target_norm_testSet.index,                       #   - DF format
#                                       columns = A1BS_target_norm_testSet.columns)
#
# A1BS_output_testSet_fileName = save_DataFrame_object(A1BS_output_testSet_DF, 'test_output' ,BS_models_save_Path)        # Save output data to a file
# A1BS_output_targetSet_fileName = save_DataFrame_object(A1BS_target_testSet_DF, 'target_output' ,BS_models_save_Path)    # Save target data to a file
#
# # ## =====================================================================================================================
# # ## ------------------------------------------------------------------------------------------------------------------- #
# # ##                   Plot results (different encoding methods by varying hidden nodes)                                 #
# # ## ------------------------------------------------------------------------------------------------------------------- #
# # ## =====================================================================================================================
# #
# BS_models_load_Path = 'C:/Users/the/Documents/Empa/Empa/PhD/Results/Approach1_BS_NN_21052018/mse_r2/'                   # Load path
#
# finalDir = ['MinMax_Label_10_log_lbfgs_5To60_adap_0.01/',                                                               #   - All inputs + Label encoding + varying hidden nodes
#             'MinMax_OneHot_10_log_lbfgs_5To60_adap_0.01/',                                                              #   - All inputs + One hot encoding + varying hidden nodes
#             'MinMax_Binary_10_log_lbfgs_5To60_adap_0.01/',                                                              #   - All inputs + Binary encoding + varying hidden nodes
#             'MinMax_10_log_lbfgs_5to60_adap_0.01_NO_CATEGORICAL/',                                                      #   - Continuous inputs + no encoding + varying hidden nodes
#             'MinMax_10_log_lbfgs_5to60_adap_0.01_NO_CATEGORICAL_AgeContinuous/']                                        #   - Continuous inputs (incl. age) + no encoding
#                                                                                                                         #                          + varying hidden nodes
#
# fileName = 'GS_obj.pkl'                                                                                                 # File name
#
# hiddenNodes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
#
# # enc_Methods = ['Label','One-Hot','Binary','No categorical (age->cat.)','No categorical (age->con.)']                    #  Encoding methods tested
#                                                                                                                           # Plot validation error for different encoding methods
# #                                                                                                                         # including variations with no categorical inputs
# #
# # gs_A1BS_output_obj1_DF = pd.DataFrame(index = np.arange(1,len(hiddenNodes)+1), columns = enc_Methods)
# # gs_A1BS_output_obj2_DF = pd.DataFrame(index = np.arange(1,len(hiddenNodes)+1), columns = enc_Methods)
# #
# # for index, directory in enumerate(finalDir):
# #     gs_A1BS_output_obj1_DF.iloc[:,index] = list(abs(load_GridSearchCV_object(BS_models_load_Path+directory,fileName)\
# #     .cv_results_['mean_test_mse'].reshape(-1,1)))                                                                       # mse test error
# #     gs_A1BS_output_obj2_DF.iloc[:, index] = list(abs(load_GridSearchCV_object(BS_models_load_Path + directory, fileName) \
# #         .cv_results_['mean_test_r2'].reshape(-1, 1)))                                                                   # r2 test error
# #
# #
# # plotData_dict = {'plot_legend': ['Label','One-Hot','Binary','No categorical (age->cat.)','No categorical (age->con.)'],
# #                  'plot_coord': {'y_1': gs_A1BS_output_obj1_DF, 'x_1': hiddenNodes,
# #                                 'y_2': gs_A1BS_output_obj2_DF, 'x_2': hiddenNodes},
# #                  'plot_labels': {'xlabel_1': 'Number of hidden neurons', 'ylabel_1': 'Mean squared error',
# #                                  'xlabel_2': 'Number of hidden neurons', 'ylabel_2': 'Coefficient of Determination ($R^2$)'},
# #                  'plot_lineSpecs': {'colorShapeMarker': ['bx-','mx-','kx-','co--','yo--'],'lineWidth': [2,2,2,1,1],
# #                                     'markerSize': [14,14,14,14,14]},
# #                  'plot_ticks': {'xticks': np.arange(0, 65, 5)}
# #                  }
# #
# # plotGridSearchComp(2, plotData_dict)
#
# labels = ['Label_test', 'One-Hot_test', 'Binary_test','No categorical_test (age->cat.)','No categorical_test (age->con.)',
#            'Label_train', 'One-Hot_train', 'Binary_train','No categorical_train (age->cat.)',
#           'No categorical_train (age->con.)']
# # Plot test compared to validation error
#                                                                                                                         # for different encoding methods
# gs_A1BS_output_obj1_DF = pd.DataFrame(index = np.arange(1,len(hiddenNodes)+1), columns = labels)
# gs_A1BS_output_obj2_DF = pd.DataFrame(index = np.arange(1,len(hiddenNodes)+1), columns = labels)
#
# for index, directory in enumerate(finalDir):
#     gs_A1BS_output_obj1_DF.iloc[:, index] = list(abs(load_GridSearchCV_object(BS_models_load_Path + directory, fileName) \
#                                                      .cv_results_['mean_train_mse'].reshape(-1, 1)))                    # mse train error
#     gs_A1BS_output_obj2_DF.iloc[:, index] = list(abs(load_GridSearchCV_object(BS_models_load_Path + directory, fileName) \
#                                                      .cv_results_['mean_train_r2'].reshape(-1, 1)))                     # r2 train error
#     gs_A1BS_output_obj1_DF.iloc[:, index+5] = list(abs(load_GridSearchCV_object(BS_models_load_Path + directory, fileName) \
#                 .cv_results_['mean_test_mse'].reshape(-1,1)))                                                           # mse test error
#     gs_A1BS_output_obj2_DF.iloc[:, index+5] = list(abs(load_GridSearchCV_object(BS_models_load_Path + directory, fileName) \
#             .cv_results_['mean_test_r2'].reshape(-1, 1)))                                                               # r2 test error
#
# plotData_dict = {'plot_legend': ['Label (test)','One-Hot (test)','Binary (test)',
#                                  'Label (train)','One-Hot (train)','Binary (train)'],
#                  'plot_coord': {'y_1': gs_A1BS_output_obj1_DF[['Label_test', 'One-Hot_test', 'Binary_test',
#                                                                'Label_train', 'One-Hot_train', 'Binary_train']],
#                                 'x_1': hiddenNodes,
#                                 'y_2':  gs_A1BS_output_obj2_DF[['Label_test', 'One-Hot_test', 'Binary_test',
#                                                                'Label_train', 'One-Hot_train', 'Binary_train']],
#                                 'x_2': hiddenNodes},
#                  'plot_labels': {'xlabel_1': 'Number of hidden neurons', 'ylabel_1': 'Mean squared error',
#                                  'xlabel_2': 'Number of hidden neurons', 'ylabel_2': 'Coefficient of Determination ($R^2$)'},
#                  'plot_lineSpecs': {'colorShapeMarker': ['bx-','mx-','kx-','bx--','mx--','kx--'],'lineWidth': [2,2,2,1,1,1],
#                                     'markerSize': [14,14,14,14,14,14]},
#                  'plot_ticks': {'xticks': np.arange(0, 65, 5)}
#                  }
#
# plotGridSearchComp(2, plotData_dict)
#
# # ## Plot results (best NN of different encoding methods (best hidden layer) - profiles)
#
# # BS_models_load_Path = 'C:/Users/the/Documents/Empa/Empa/PhD/Results/Approach1_BS_NN_21052018/mse/'
# # finalDir = ['MinMax_Label_10_log_lbfgs_10To100_adap_0.01/','MinMax_OneHot_10_log_lbfgs_10To100_adap_0.01/','MinMax_Binary_10_log_lbfgs_10To100_adap_0.01/']
#
# A1BS_output_testSet_label = load_DataFrame_object(BS_models_load_Path+finalDir[0], 'dataFrame_test_output.pkl')
# A1BS_output_testSet_oneHot = load_DataFrame_object(BS_models_load_Path+finalDir[1], 'dataFrame_test_output.pkl')
# A1BS_output_testSet_Binary = load_DataFrame_object(BS_models_load_Path+finalDir[2], 'dataFrame_test_output.pkl')
# A1BS_output_testSet_NoCat = load_DataFrame_object(BS_models_load_Path+finalDir[3], 'dataFrame_test_output.pkl')
#
# A1BS_target_testSet = load_DataFrame_object(BS_models_load_Path+finalDir[0], 'dataFrame_target_output.pkl')
#
# total_area_testSet = buildinfInfo_DF.loc[testSet_ids+1,'Total Area']
#
# # A1BS_output_testSet_oneHot_FGS = load_DataFrame_object(BS_models_load_Path+finalDir[3], 'dataFrame_test_output.pkl')
# #
# # # A1BS_target_testSet1 = load_DataFrame_object(BS_models_load_Path+finalDir[1], 'dataFrame_target_output.pkl')
# A1BS_target_testSet_normPerSM = np.divide(A1BS_target_testSet['AH_Base case'],total_area_testSet)
# #
# A1BS_target_testSet_normPerSM = np.divide(A1BS_target_testSet['AH_Base case'],total_area_testSet)
# A1BS_output_testSet_label_normPerSM = np.divide(A1BS_output_testSet_label['AH_Base case'],total_area_testSet)
# A1BS_output_testSet_oneHot_normPerSM = np.divide(A1BS_output_testSet_oneHot['AH_Base case'],total_area_testSet)
# A1BS_output_testSet_NoCat = np.divide(A1BS_output_testSet_NoCat['AH_Base case'],total_area_testSet)
#
# # A1BS_output_testSet_Binary_normPerSM = np.divide(A1BS_output_testSet_Binary['PH_Base case'],total_area_testSet)
# # A1BS_output_testSet_oneHot_FGS_normPerSM = np.divide(A1BS_output_testSet_oneHot_FGS['AH_Base case'],total_area_testSet)
# #
# plt.figure(figsize=(20,10))
# x_axis = np.arange(1,len(testSet_ids)+1)
# p1, = plt.plot(x_axis,A1BS_target_testSet_normPerSM,'g.--',linewidth=3,markersize=10, label = 'Target')
# p2, = plt.plot(x_axis,A1BS_output_testSet_label_normPerSM,'b.-',linewidth=0.5,markersize=10, label = 'Label')
# p3, = plt.plot(x_axis,A1BS_output_testSet_oneHot_normPerSM,'m.-',linewidth=0.5,markersize=10, label = 'One-Hot')
# p5, = plt.plot(x_axis,A1BS_output_testSet_NoCat,'r.-',linewidth=0.5,markersize=10, label = 'No enc.')
# plt.legend(handles=[p1,p2,p3, p5])
# plt.ylim((0,400))
# plt.xlim((780, 800))
# font = {'family' : 'normal',
#         'weight': 'normal',
#         'size'   : 18}
# plt.ylabel('Annual Heating Demand (kWh/$m^2$/a)', fontsize=18)
# plt.xlabel('Buildings', fontsize=18)
# plt.rc('font', **font)
# plt.grid()
# plt.show()
#
# a = load_GridSearchCV_object(BS_models_load_Path + finalDir[1], fileName)
# b = a.best_estimator_.predict(A1BS_input_testSet)
# #
# # # p2, = plt.plot(x_axis,A1BS_output_testSet_label_normPerSM,'b.-',linewidth=0.5,markersize=10, label = 'Label')
# # p3, = plt.plot(x_axis,A1BS_output_testSet_oneHot_normPerSM,'m.-',linewidth=0.5,markersize=10, label = 'One-Hot')
# # p4, = plt.plot(x_axis,A1BS_output_testSet_oneHot_FGS_normPerSM,'m.-',linewidth=2, label = 'One-Hot (Full GS)')
# # # p5, = plt.plot(x_axis,A1BS_output_testSet_Binary_normPerSM,'k.-',linewidth=0.5,markersize=10, label = 'Binary')
# # plt.xlim((224, 228))
# # plt.ylim((0,250))
# # plt.xlabel('Buildings', fontsize=18)
# # plt.ylabel('Annual Heating Demand (kWh/$m^2$/a)', fontsize=18)
# # plt.xticks(np.arange(224, 228,1))
# # plt.legend(handles=[p1, p3, p4]) #, p2, p3, p4, p5])
# # font = {'family' : 'normal',
# #         'weight': 'normal',
# #         'size'   : 18}
# # plt.rc('font', **font)
# # plt.grid()
# # plt.show()
# #/ #
# #
# # # plt.figure(figsize=(20,10))
# # # # p1, = plt.plot(x_axis,A1BS_target_testSet['AH_Base case']/total_area_testSet,'g--',linewidth=3, label = 'Target')
# # # # p1, = plt.plot(np.arange(1,ED_annual_h_DF.shape[0]+1),ED_annual_h_DF['AH_Base case']/buildinfInfo_DF['Total Area'],'g--',linewidth=3, label = 'Target')
# # # # p1, = plt.plot(np.arange(1,len(testSet_ids)+1),A1BS_output_testSet_label_normPerSM)
# # # # p1, = plt.plot(np.arange(1,len(testSet_ids)+1),A1BS_output_testSet_oneHot_FGS_normPerSM)
# # # # p1, = plt.plot(np.arange(1,len(testSet_ids)+1),A1BS_output_testSet_oneHot_FGS['AH_Base case'])
# # # # plt.plot(np.arange(1,len(testSet_ids)+1),A1BS_target_testSet['AH_Base case'])
# # # # plt.plot(np.arange(1,len(testSet_ids)+1),A1BS_output_testSet_label['AH_Base case'])
# # # # plt.plot(np.arange(1,len(testSet_ids)+1),A1BS_output_testSet_oneHot['AH_Base case'])
# # # # plt.plot(np.arange(1,len(testSet_ids)+1),A1BS_output_testSet_Binary['AH_Base case'])
# # # plt.show()
# #
# #
# # print('... Successful')

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                       Approach 1 - Energy Hub ML algorithm - 1.Cost&Emissions                       #
## ------------------------------------------------------------------------------------------------------------------- #
## =====================================================================================================================


print("--* Approach 1 - Energy hub ML training *--")

A1ΕΗ1_input = pd.concat([profiles_norm_DF, buildingInfo_norm_DF[['Roof slope','Roof orientation']]                      # Input = [ Annual heating demand; Annual electricity demand;
                        ,input_catData_enc_DF,buildingInfo_norm_DF[['Ground floor area','Height']]], axis=1)            #     heating peak; electricity peak; roof inclination;
                                                                                                                        #     roof orientation; Building age; heating energy carrier;
                                                                                                                        #     DHW energy carrier; building type; floor area;
                                                                                                                        #     Building height]

A1ΕΗ1_input_trainSet = A1ΕΗ1_input.iloc[trainSet_ids,:]                                                                 #   - Separate input to training
A1ΕΗ1_input_testSet = A1ΕΗ1_input.iloc[testSet_ids,:]                                                                   #     and test set

# A1ΕΗ1_target = EH_output_contData_norm_DF.loc[:,['Cost','Emissions']]                                                           # Target = [ Cost; Emissions ]                                                          #     and test set

A1ΕΗ1_con_target = EH_output_objectiveData_norm_DF #EH_output_contData_norm_DF[['Cost','Emissions']].copy()             # Target = [ cost, emissions ]
A1ΕΗ1_con_target.index = np.arange(0,A1ΕΗ1_con_target.shape[0])

A1ΕΗ_con_target_DF = pd.DataFrame()                                                                                     # Process target format to have in total 20 outputs
output_labels = np.concatenate([['cost_PFP_' + str(pfp) for pfp in np.arange(1,11)],
                                ['emissions_PFP_' + str(pfp) for pfp in np.arange(1,11)]])
countPF = 0
for pf_num in range(0,int(A1ΕΗ1_con_target.shape[0]/10)):
    pf = A1ΕΗ1_con_target.iloc[0+10*pf_num:10+10*pf_num, :]
    pf_transp = pf.transpose()
    pf_array = np.array(pf_transp)
    pf_array_flat = pf_array.flatten()[np.newaxis]                                                                      #   - Adding one more dimension (from 1D to 2D)
    temp_target = pd.DataFrame(pf_array_flat, columns = output_labels)
    A1ΕΗ_con_target_DF = pd.concat([A1ΕΗ_con_target_DF, temp_target])
A1ΕΗ_con_target_DF.index = np.arange(1,int(A1ΕΗ1_con_target.shape[0]/10)+1)                                             #   - matrix of [12802 x 20]

A1ΕΗ1_target_norm_trainSet = A1ΕΗ_con_target_DF.iloc[trainSet_ids,:]                                                    #   - Separate target to training
A1ΕΗ1_target_norm_testSet = A1ΕΗ_con_target_DF.iloc[testSet_ids,:]


# checkPointPath = 'C:/Users/the/Documents/Empa/Empa/PhD/Results/Approach1_BS_NN_21052018/keras/run_13062018/'
#
# random_gen_seed = 4
#
# (keras_EH_gs_1, gs_history) = kerasNN(A1ΕΗ1_input_trainSet,A1ΕΗ1_target_norm_trainSet, True, random_gen_seed, checkPointPath)
#
# plt.figure(figsize=(20, 10))
# plt.plot(np.arange(10,206,10),abs(keras_EH_gs_1.cv_results_['mean_train_score']))
# plt.plot(np.arange(10,206,10),abs(keras_EH_gs_1.cv_results_['mean_test_score']))
# plt.title('model performance (label encoding)')
# plt.ylabel('mse', fontsize=18)
# plt.xlabel('hidden neurons', fontsize=18)
# plt.legend(['train', 'validation'], loc= 'center right')#'upper right')
# font = {'family': 'normal',
#         'weight': 'normal',
#         'size': 18}
# plt.rc('font', **font)
# plt.grid()
# plt.show()
#
# plt.figure(figsize=(20, 10))
# plt.plot(keras_EH_gs_1.best_estimator_.model.model.history.history['mean_squared_error'])                                                 # summarize history for accuracy
# plt.plot(keras_EH_gs_1.best_estimator_.model.model.history.history['val_mean_squared_error'])
# plt.title('model performance')
# plt.ylabel('mse', fontsize=18)
# plt.xlabel('epoch', fontsize=18)
# plt.legend(['train', 'validation'], loc='upper right')
# font = {'family': 'normal',
#         'weight': 'normal',
#         'size': 18}
# plt.rc('font', **font)
# plt.grid()
# plt.show()
#
# plt.plot(keras_EH_gs_1.best_estimator_.model.model.history.history['loss'])                                                # summarize history for loss
# plt.plot(keras_EH_gs_1.best_estimator_.model.model.history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss', fontsize=18)
# plt.xlabel('epoch', fontsize=18)
# plt.legend(['train', 'validation'], loc='upper right')
# font = {'family': 'normal',
#         'weight': 'normal',
#         'size': 18}
# plt.rc('font', **font)
# plt.grid()
# plt.show()
#
#
#
#
# A1EH1_norm_output_testSet = keras_EH_gs_1.best_estimator_.predict(A1ΕΗ1_input_testSet)                                             # Prediction (on test set) with best estimator
# A1EH1_norm_output_testSet_DF = pd.DataFrame(A1EH1_norm_output_testSet, index = A1ΕΗ1_target_norm_testSet.index,         #   - DF format
#                                            columns = A1ΕΗ1_target_norm_testSet.columns)
#
# A1EH1_norm_output_testSet_np = np.zeros([A1EH1_norm_output_testSet_DF.shape[0]*10,2])                                   # Reverse output test set to initial format to apply inverse
# A1ΕΗ1_norm_target_testSet_np = np.zeros([A1ΕΗ1_target_norm_testSet.shape[0]*10,2])
# count = 0
# for testSample,targetSample in zip(A1EH1_norm_output_testSet_DF.iterrows(),A1ΕΗ1_target_norm_testSet.iterrows()):
#     A1EH1_norm_output_testSet_np[0+10*count:10+10*count, :] = testSample[1].reshape(-1,10).transpose()
#     A1ΕΗ1_norm_target_testSet_np[0+10*count:10+10*count, :] = targetSample[1].reshape(-1,10).transpose()
#     count += 1
# A1EH1_norm_output_testSet_resDF = pd.DataFrame(A1EH1_norm_output_testSet_np, columns= ['Cost','Emissions'])             #   - Matrix [19200x2]
# A1ΕΗ1_norm_target_testSet_resDF = pd.DataFrame(A1ΕΗ1_norm_target_testSet_np, columns= ['Cost','Emissions'])
#
# A1EH1_output_testSet = EH_output_objectiveData_info.inverse_transform(A1EH1_norm_output_testSet_resDF)                  # Inverse normalization process for output data
# A1EH1_output_testSet_DF = pd.DataFrame(A1EH1_output_testSet,columns = ['Cost','Emissions'])                             #   - DF format
#
#
# A1EH1_target_testSet = EH_output_objectiveData_info.inverse_transform(A1ΕΗ1_norm_target_testSet_resDF)                  # Inverse normalization process for respective target data
# A1EH1_target_testSet_DF = pd.DataFrame(A1EH1_target_testSet,columns = ['Cost','Emissions'])                             #   - DF format

#
# gs = MLPerceptron(A1ΕΗ1_input_trainSet,A1ΕΗ1_target_norm_trainSet, 'Regression', 'True', 'hypervolume')                 # Train ML
#
#
# finalDir = ['MinMax_Label_10_log_lbfgs_5To60_adap_0.01/','MinMax_OneHot_10_log_lbfgs_5To60_adap_0.01/',
#         'MinMax_Binary_10_log_lbfgs_5To60_adap_0.01/','MinMax_onehot_10_logtanh_lbfgsadam_5To50_adap_0.010.10110100/']
#
# EH_models_save_Path = 'C:/Users/the/Documents/Empa/Empa/PhD/Results/Approach1_EH_NN_10052018/continuous/'+finalDir[2]   # Load path
#
# fileName = save_GridSearchCV_object(gs,EH_models_save_Path)                                                             # Save training grid search results
#
# A1EH1_norm_output_testSet = gs.best_estimator_.predict(A1ΕΗ1_input_testSet)                                             # Prediction (on test set) with best estimator
# A1EH1_norm_output_testSet_DF = pd.DataFrame(A1EH1_norm_output_testSet, index = A1ΕΗ1_target_norm_testSet.index,         #   - DF format
#                                            columns = A1ΕΗ1_target_norm_testSet.columns)
#
# A1EH1_norm_output_testSet_np = np.zeros([A1EH1_norm_output_testSet_DF.shape[0]*10,2])                                   # Reverse output test set to initial format to apply inverse
# A1ΕΗ1_norm_target_testSet_np = np.zeros([A1ΕΗ1_target_norm_testSet.shape[0]*10,2])
# count = 0
# for testSample,targetSample in zip(A1EH1_norm_output_testSet_DF.iterrows(),A1ΕΗ1_target_norm_testSet.iterrows()):
#     A1EH1_norm_output_testSet_np[0+10*count:10+10*count, :] = testSample[1].reshape(-1,10).transpose()
#     A1ΕΗ1_norm_target_testSet_np[0+10*count:10+10*count, :] = targetSample[1].reshape(-1,10).transpose()
#     count += 1
# A1EH1_norm_output_testSet_resDF = pd.DataFrame(A1EH1_norm_output_testSet_np, columns= ['Cost','Emissions'])             #   - Matrix [19200x2]
# A1ΕΗ1_norm_target_testSet_resDF = pd.DataFrame(A1ΕΗ1_norm_target_testSet_np, columns= ['Cost','Emissions'])
#
# A1EH1_output_testSet = EH_output_objectiveData_info.inverse_transform(A1EH1_norm_output_testSet_resDF)                  # Inverse normalization process for output data
# A1EH1_output_testSet_DF = pd.DataFrame(A1EH1_output_testSet,columns = ['Cost','Emissions'])                             #   - DF format
#
#
# A1EH1_target_testSet = EH_output_objectiveData_info.inverse_transform(A1ΕΗ1_norm_target_testSet_resDF)                  # Inverse normalization process for respective target data
# A1EH1_target_testSet_DF = pd.DataFrame(A1EH1_target_testSet,columns = ['Cost','Emissions'])                             #   - DF format
#
#
# A1EH1_output_testSet_fileName = save_DataFrame_object(A1EH1_output_testSet_DF, 'test_output' ,EH_models_save_Path)      # Save output data to a file
# A1EH1_output_targetSet_fileName = save_DataFrame_object(A1EH1_target_testSet_DF, 'target_output' ,EH_models_save_Path)  # Save target data to a file
#
# ## =====================================================================================================================
# ## ------------------------------------------------------------------------------------------------------------------- #
# ##                   Plot results (different encoding methods by varying hidden nodes)                                 #
# ## ------------------------------------------------------------------------------------------------------------------- #
# ## =====================================================================================================================
#
# EH_models_load_Path = 'C:/Users/the/Documents/Empa/Empa/PhD/Results/Approach1_EH_NN_10052018/continuous/'               # Load path
#
# finalDir = ['MinMax_Label_10_log_lbfgs_5To60_adap_0.01/','MinMax_OneHot_10_log_lbfgs_5To60_adap_0.01/',
#         'MinMax_Binary_10_log_lbfgs_5To60_adap_0.01/']
#
# fileName = 'GS_obj.pkl'                                                                                                 # File name
#
# enc_Methods = ['Label','One-Hot','Binary']                                                                               # Encoding methods tested
#
# gs_A1EH1_output_DF = pd.DataFrame(columns = enc_Methods)
#
# gs_A1EH1_output = np.zeros([2,3])
#
# # for index, directory in enumerate(finalDir):
# #     gs_A1EH1_output[:,index] = load_GridSearchCV_object(EH_models_load_Path+directory,fileName).cv_results_['mean_test_score'].reshape(-1,1)
#
# # gs_A1EH1_output_label = load_GridSearchCV_object(EH_models_load_Path+finalDir[0], fileName)                             # Load files
# # gs_A1EH1_output_oneHot = load_GridSearchCV_object(EH_models_load_Path+finalDir[1], fileName)
# # gs_A1EH1_output_Binary = load_GridSearchCV_object(EH_models_load_Path+finalDir[2], fileName)
# # # gs_A1BS_output_oneHot_FullGS = load_GridSearchCV_object(EH_models_load_Path+finalDir[3], fileName)
# #
# # A1EH1_label_trainScore = abs(gs_A1EH1_output_label.cv_results_['mean_train_score'])                                     # Training performance for single-criteria GS
# # A1EH1_oneHot_trainScore = abs(gs_A1EH1_output_label.cv_results_['mean_train_score'])
# # A1EH1_Binary_trainScore = abs(gs_A1EH1_output_label.cv_results_['mean_train_score'])
# #
# # A1EH1_label_testScore = abs(gs_A1EH1_output_label.cv_results_['mean_test_score'])                                       # Validation performance for single-criteria GS
# # A1EH1_oneHot_testScore = abs(gs_A1EH1_output_oneHot.cv_results_['mean_test_score'])
# # A1EH1_Binary_testScore = abs(gs_A1EH1_output_Binary.cv_results_['mean_test_score'])
#
# hiddenNodes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
#
# labels = ['Label_test', 'One-Hot_test', 'Binary_test',
#           'Label_train', 'One-Hot_train', 'Binary_train']
#
# # Plot test compared to validation error
#                                                                                                                         # for different encoding methods
# gs_A1EH1_output_obj1_DF = pd.DataFrame(index = np.arange(1,len(hiddenNodes)+1), columns = labels)
#
# for index, directory in enumerate(finalDir):
#     gs_A1EH1_output_obj1_DF.iloc[:, index] = list(abs(load_GridSearchCV_object(EH_models_load_Path + directory, fileName) \
#                                                      .cv_results_['mean_train_score'].reshape(-1, 1)))                  # train error (ridge regression loss error)
#     gs_A1EH1_output_obj1_DF.iloc[:, index+3] = list(abs(load_GridSearchCV_object(EH_models_load_Path + directory, fileName) \
#                 .cv_results_['mean_test_score'].reshape(-1,1)))                                                         # test error  (Hypervolume difference index)
#
#
# plotData_dict = {'plot_legend': ['Label (test - mse+reg)','One-Hot (test - mse+reg)','Binary (test - mse+reg)',
#                                  'Label (train - Hypervolume)','One-Hot (train - Hypervolume)','Binary (train - Hypervolume)'],
#                  'plot_coord': {'y_1': gs_A1EH1_output_obj1_DF[['Label_test', 'One-Hot_test', 'Binary_test',
#                                                                'Label_train', 'One-Hot_train', 'Binary_train']],
#                                 'x_1': hiddenNodes},
#                  'plot_labels': {'xlabel_1': 'Number of hidden neurons', 'ylabel_1': 'Error'},
#                  'plot_lineSpecs': {'colorShapeMarker': ['bx-','mx-','kx-','bx--','mx--','kx--'],'lineWidth': [2,2,2,1,1,1],
#                                     'markerSize': [14,14,14,14,14,14]},
#                  'plot_ticks': {'xticks': np.arange(0, 65, 5)}
#                  }
#
# plotGridSearchComp(1, plotData_dict)
#
#
#
#
#
#
#
#
#
#
# #
# # plt.figure(figsize=(20, 10))
# # p1, = plt.plot(hiddenNodes, A1EH1_label_testScore, 'bx-', linewidth=2, markersize=14, label='Label')
# # p2, = plt.plot(hiddenNodes, A1EH1_oneHot_testScore, 'mx-', linewidth=2, markersize=14, label='One-Hot')
# # p3, = plt.plot(hiddenNodes, A1EH1_Binary_testScore, 'kx-', linewidth=2, markersize=14, label='Binary')
# # plt.xlim((0, 65))
# # plt.xlabel('Number of hidden neurons', fontsize=18)
# # plt.ylabel('Hypervolume difference Indicator', fontsize=18)
# # plt.xticks(np.arange(0, 65, 5))
# # plt.legend(handles=[p1, p2, p3])
# # plt.grid()
# # font = {'family': 'normal',
# # #         'weight': 'normal',
# # #         'size': 18}
# # # plt.rc('font', **font)
# # # plt.show()
# #
#
# # Plot results (best NN of different encoding methods (best hidden layer) - profiles)
#
# # EH_models_load_Path = 'C:/Users/the/Documents/Empa/Empa/PhD/Results/Approach1_EH_NN_10052018/continuous/'               # Load path
# #
# # finalDir = ['MinMax_Label_10_log_lbfgs_5To60_adap_0.01/','MinMax_OneHot_10_log_lbfgs_5To60_adap_0.01/',
# #         'MinMax_Binary_10_log_lbfgs_5To60_adap_0.01/']
# #
# # A1EH1_output_testSet_label = load_DataFrame_object(EH_models_load_Path+finalDir[0], 'dataFrame_test_output.pkl')
# # A1EH1_output_testSet_oneHot = load_DataFrame_object(EH_models_load_Path+finalDir[1], 'dataFrame_test_output.pkl')
# # A1EH1_output_testSet_Binary = load_DataFrame_object(EH_models_load_Path+finalDir[2], 'dataFrame_test_output.pkl')
# #
# # A1EH1_target_testSet = load_DataFrame_object(EH_models_load_Path+finalDir[0], 'dataFrame_target_output.pkl')
# #
# total_area_testSet = buildinfInfo_DF.loc[testSet_ids+1,'Total Area']
# A1EH1_target_testSet = A1EH1_target_testSet_DF
#
# # A1EH1_output_testSet_oneHot_FGS = load_DataFrame_object(BS_models_load_Path+finalDir[3], 'dataFrame_test_output.pkl')
# #
# # # A1BS_target_testSet1 = load_DataFrame_object(BS_models_load_Path+finalDir[1], 'dataFrame_target_output.pkl')
# # # A1BS_target_testSet_normPerSM1 = np.divide(A1BS_target_testSet1['AH_Base case'],total_area_testSet)
# #
#
# A1EH1_output_testSet_label = A1EH1_output_testSet_DF
#
#
# A1EH1_target_testSet_normPerSM = pd.DataFrame(index = A1EH1_target_testSet.index, columns = A1EH1_target_testSet.columns)
# A1EH1_output_testSet_label_normPerSM = pd.DataFrame(index = A1EH1_target_testSet.index, columns = A1EH1_target_testSet.columns)
# A1EH1_output_testSet_oneHot_normPerSM = pd.DataFrame(index = A1EH1_target_testSet.index, columns = A1EH1_target_testSet.columns)
# A1EH1_output_testSet_Binary_normPerSM = pd.DataFrame(index = A1EH1_target_testSet.index, columns = A1EH1_target_testSet.columns)
#
#
# for index,buildingArea in enumerate(total_area_testSet):
#     A1EH1_target_testSet_normPerSM.loc[index * 10:(index + 1) * 10 - 1,'Cost'] = np.divide(A1EH1_target_testSet.loc[index * 10:(index + 1) * 10 - 1,'Cost'],buildingArea)
#     A1EH1_output_testSet_label_normPerSM.loc[index * 10:(index + 1) * 10 - 1,'Cost'] = np.divide(A1EH1_output_testSet_label.loc[index * 10:(index + 1) * 10 - 1,'Cost'],buildingArea)
#     # A1EH1_output_testSet_oneHot_normPerSM.loc[index * 10:(index + 1) * 10 - 1,'Cost'] = np.divide(A1EH1_output_testSet_oneHot.loc[index * 10:(index + 1) * 10 - 1,'Cost'],buildingArea)
#     # A1EH1_output_testSet_Binary_normPerSM.loc[index * 10:(index + 1) * 10 - 1,'Cost'] = np.divide(A1EH1_output_testSet_Binary.loc[index * 10:(index + 1) * 10 - 1,'Cost'],buildingArea)
#
#     A1EH1_target_testSet_normPerSM.loc[index * 10:(index + 1) * 10 - 1, 'Emissions'] = np.divide(
#         A1EH1_target_testSet.loc[index * 10:(index + 1) * 10, 'Emissions'], buildingArea)
#     A1EH1_output_testSet_label_normPerSM.loc[index * 10:(index + 1) * 10 - 1, 'Emissions'] = np.divide(
#         A1EH1_output_testSet_label.loc[index * 10:(index + 1) * 10, 'Emissions'], buildingArea)
#     # A1EH1_output_testSet_oneHot_normPerSM.loc[index * 10:(index + 1) * 10 - 1, 'Emissions'] = np.divide(
#     #     A1EH1_output_testSet_oneHot.loc[index * 10:(index + 1) * 10, 'Emissions'], buildingArea)
#     # A1EH1_output_testSet_Binary_normPerSM.loc[index * 10:(index + 1) * 10 - 1, 'Emissions'] = np.divide(
#     #     A1EH1_output_testSet_Binary.loc[index * 10:(index + 1) * 10, 'Emissions'], buildingArea)
#
#
# # plots_ids = [44,50,72,83]                                                                                               # quite good results
# plots_ids = [82,92,94,99]                                                                                               # quite bad results
# # plots_ids = [80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]
#
# plt.figure(figsize=(20,10))
# for count,index in enumerate(plots_ids):
#     plt.subplot(2,2,count+1)
#     p1, = plt.plot(A1EH1_target_testSet_normPerSM.loc[index * 10:(index + 1) * 10 - 1,'Cost'],
#                    A1EH1_target_testSet_normPerSM.loc[index * 10:(index + 1) * 10 - 1, 'Emissions'],
#                    'gx--', linewidth=3, markersize=10, label = 'Target')
#     p2, = plt.plot(A1EH1_output_testSet_label_normPerSM.loc[index * 10:(index + 1) * 10 - 1, 'Cost'],
#                    A1EH1_output_testSet_label_normPerSM.loc[index * 10:(index + 1) * 10 - 1, 'Emissions'],
#                    'b.-', linewidth=0.5, markersize=10, label='Label')
#     # p3, = plt.plot(A1EH1_output_testSet_oneHot_normPerSM.loc[index * 10:(index + 1) * 10 - 1, 'Cost'],
#     #                A1EH1_output_testSet_oneHot_normPerSM.loc[index * 10:(index + 1) * 10 - 1, 'Emissions'],
#     #                'm.-', linewidth=0.5, markersize=10, label='One-Hot')
#     # p4, = plt.plot(A1EH1_output_testSet_Binary_normPerSM.loc[index * 10:(index + 1) * 10 - 1, 'Cost'],
#     #                A1EH1_output_testSet_Binary_normPerSM.loc[index * 10:(index + 1) * 10 - 1, 'Emissions'],
#     #                'k.-', linewidth=0.5, markersize=10, label='Binary')
#     plt.xlabel('Cost [CHF/$m^2$/a]', fontsize=18)
#     plt.ylabel('GHG Emissions [kg $CO_2$/$m^2$/a]', fontsize=18)
#     # plt.legend(handles=[p1, p2, p3, p4])
#     plt.legend(handles=[p1, p2])
#     plt.grid()
# font = {'family': 'normal',
#             'weight': 'normal',
#             'size': 18}
# plt.rc('font', **font)
# plt.show()
#
#
#
#
# ## =====================================================================================================================
# ## ------------------------------------------------------------------------------------------------------------------- #
# ##                            Approach 1 - Energy Hub ML algorithm - 2.Retrofit&System Selection                       #     <----- UNDER IMPLEMENTATION
# ## ------------------------------------------------------------------------------------------------------------------- #
# ## =====================================================================================================================

print("--* Approach 1 - Energy hub ML training *--")

A1ΕΗ2_input = pd.concat([profiles_norm_DF, buildingInfo_norm_DF[['Roof slope','Roof orientation']]                      # Input = [ Annual heating demand; Annual electricity demand;
                        ,input_catData_enc_DF,buildingInfo_norm_DF[['Ground floor area','Height']]], axis=1)            #     heating peak; electricity peak; roof inclination;
                                                                                                                        #     roof orientation; Building age; heating energy carrier;
                                                                                                                        #     DHW energy carrier; building type; floor area;
                                                                                                                        #     Building height]

A1ΕΗ2_input_trainSet = A1ΕΗ2_input.iloc[trainSet_ids,:]                                                                 #   - Separate input to training
A1ΕΗ2_input_testSet = A1ΕΗ2_input.iloc[testSet_ids,:]                                                                   #     and test set

cat_target = output_catData_enc_DF.copy()                                                                               # Target = [retrofit, system selection]
cat_target.index = np.arange(0,cat_target.shape[0])

A1ΕΗ2_cat_target = pd.DataFrame()                                                                                       # Process target format to have in total 20 outputs
output_labels = np.concatenate([['ret.Sel._PFP_' + str(pfp) for pfp in np.arange(1,11)],                                #    - HAS TO BE CHECKED FOR ONE-HOT & BINARY!!!!
                                ['sys.Sel._PFP_' + str(pfp) for pfp in np.arange(1,11)]])

countPF = 0
for pf_num in range(0,int(cat_target.shape[0]/10)):
    pf = cat_target.iloc[0+10*pf_num:10+10*pf_num, :]
    pf_transp = pf.transpose()
    pf_array = np.array(pf_transp)
    pf_array_flat = pf_array.flatten()[np.newaxis] # Adding one more dimension (from 1D to 2D)
    temp_target = pd.DataFrame(pf_array_flat, columns = output_labels)
    A1ΕΗ2_cat_target = pd.concat([A1ΕΗ2_cat_target, temp_target])
A1ΕΗ2_cat_target.index = np.arange(1,int(cat_target.shape[0]/10)+1)                                                     # matrix of [12802 x 20] - ONLY FOR LABEL ENCODING

A1ΕΗ2_target_norm_trainSet = A1ΕΗ2_cat_target.iloc[trainSet_ids,:]                                                      #   - Separate target to training
A1ΕΗ2_target_norm_testSet = A1ΕΗ2_cat_target.iloc[testSet_ids,:]
# #
# # # gs = MLPerceptron(A1ΕΗ2_input_trainSet,A1ΕΗ2_target_norm_trainSet, 'Classification', 'True', 'hypervolume')             # Train ML
# #
# #
## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                Approach 1 - Energy Hub ML algorithm - 3.Cost,Emissions,Retrofit&System Selection                    #
## ------------------------------------------------------------------------------------------------------------------- #
## =====================================================================================================================

print("--* Approach 1 - Energy hub ML training *--")

A1ΕΗ3_input = pd.concat([profiles_norm_DF,
                         # [['AH_Base case', 'AH_Full ret.', 'AH_Wall ret.', 'AH_Window ret.',
                         #                    'AH_Wall & Window ret.', 'AH_Roof ret.', 'AH_Wall & Win & Roof ret.',
                         #                    'AE_Base case','PH_Base case', 'PH_Full ret.', 'PH_Wall ret.', 'PH_Window ret.',
                         #                    'PH_Wall & Window ret.', 'PH_Roof ret.', 'PH_Wall & Win & Roof ret.',
                         #                    'PE_Base case']],
                         buildingInfo_norm_DF[['Roof slope','Roof orientation']]                      # Input = [ Annual heating demand; Annual electricity demand;
                        ,input_catData_enc_DF[['Age', 'Heating en.Carrier', 'DHW en.Carrier']],
                         # [['Age', 'Heating en.Carrier', 'DHW en.Carrier']],                           # EXCLUDE BUILDING TYPE
                         buildingInfo_norm_DF[['Ground floor area','Height']]], axis=1)                                 #     heating peak; electricity peak; roof inclination;
                                                                                                                        #     roof orientation; Building age; heating energy carrier;
                                                                                                                        #     DHW energy carrier; building type; floor area;
                                                                                                                        #     Building height]

A1ΕΗ3_input_trainSet = A1ΕΗ3_input.iloc[trainSet_ids,:]                                                                 #   - Separate input to training
A1ΕΗ3_input_testSet = A1ΕΗ3_input.iloc[testSet_ids,:]                                                                   #     and test set

rf_classifier = hybridMLAlgorithm(A1ΕΗ3_input_trainSet,A1ΕΗ1_target_norm_trainSet,A1ΕΗ2_target_norm_trainSet,'False')

rf_classifier_1.oob_score_

fi = pd.DataFrame(rf_classifier_1.feature_importances_)
fi_st_info = fi.describe()
fi_mean = [fi_st_info.loc['mean'][0] for i in np.arange(0,74,1)]
fi_25percentile = [fi_st_info.loc['25%'][0] for i in np.arange(0,74,1)]
fi_75percentile = [fi_st_info.loc['75%'][0] for i in np.arange(0,74,1)]

plt.figure(figsize=(14, 10))
plt.bar(np.arange(1,73,2),rf_classifier_1.feature_importances_)
p1, = plt.plot(np.arange(0,74,1),np.transpose(fi_mean), 'c-', linewidth=2, label='Mean')
p2, = plt.plot(np.arange(0,74,1),np.transpose(fi_25percentile),'r--', linewidth=2, label='25% Percentile')
p3, = plt.plot(np.arange(0,74,1),np.transpose(fi_75percentile),'g--', linewidth=2, label='75% Percentile')
plt.xticks(np.arange(1,73,2),('AH_BC', 'AH_FR', 'AH_WaR', 'AH_WinR','AH_WaWinR', 'AH_RR','AH_WaWinRR',
                              'AE_BC', 'AE_FR', 'AE_WaR', 'AE_WinR','AE_WaWinR', 'AE_RR', 'AE_WaWinRR',
                              'PH_BC', 'PH_FR', 'PH_WaR', 'PH_WinR','PH_WaWinR', 'PH_RR', 'PH_WaWinRR',
                              'PE_BC', 'PE_FR', 'PE_WaR', 'PE_WinR','PE_WaWinR', 'PE_RR', 'PE_WaWinRR',
                              'R.slope', 'R.orientation','Age', 'En.Carrier(heat)', 'En.Carrier(DHW)',
                              'Type','GFA','Height'), rotation = 45, horizontalalignment='right')
# plt.bar(np.arange(1,71,2),rf_classifier_1.feature_importances_)
# p1, = plt.plot(np.arange(0,74,1),np.transpose(fi_mean), 'c-', linewidth=2, label='Mean')
# p2, = plt.plot(np.arange(0,74,1),np.transpose(fi_25percentile),'r--', linewidth=2, label='25% Percentile')
# p3, = plt.plot(np.arange(0,74,1),np.transpose(fi_75percentile),'g--', linewidth=2, label='75% Percentile')
# plt.xticks(np.arange(1,71,2),('AH_BC', 'AH_FR', 'AH_WaR', 'AH_WinR','AH_WaWinR', 'AH_RR','AH_WaWinRR',
#                               'AE_BC', 'AE_FR', 'AE_WaR', 'AE_WinR','AE_WaWinR', 'AE_RR', 'AE_WaWinRR',
#                               'PH_BC', 'PH_FR', 'PH_WaR', 'PH_WinR','PH_WaWinR', 'PH_RR', 'PH_WaWinRR',
#                               'PE_BC', 'PE_FR', 'PE_WaR', 'PE_WinR','PE_WaWinR', 'PE_RR', 'PE_WaWinRR',
#                               'R.slope', 'R.orientation','Age', 'En.Carrier(heat)', 'En.Carrier(DHW)',
#                               'GFA','Height'), rotation = 45, horizontalalignment='right')
plt.ylabel('Feature importance', fontsize=15)
plt.legend(handles=[p2, p1, p3])
# plt.grid()
font = {'family': 'normal',
            'weight': 'normal',
            'size': 10}
plt.rc('font', **font)
plt.show()



# ## =====================================================================================================================
# ## ------------------------------------------------------------------------------------------------------------------- #
# ##                                 Plot results (randomly selected parameters)                                         #
# ## ------------------------------------------------------------------------------------------------------------------- #
# ## =====================================================================================================================
#
#
# hiddenLayers = [10,30,50,80,120]
# depthOfTrees = [5,10]
# NumOfTrees = 10
# hypVol_depth5 = [0.0163, 0.0183, 0.0189, 0.0193, 0.0203]
# overSSR_depth5 = [0.57, 0.585, 0.6, 0.6, 0.59]
# overSSR_ret_depth5 = [0.54, 0.54, 0.55, 0.55, 0.54]
# overSSR_sys_depth5 = [0.6, 0.63, 0.64, 0.65, 0.64]
#
# hypVol_depth10 = [0.0160, 0.0187, 0.0191, 0.0193, 0.0200]
# overSSR_depth10 = [0.57, 0.59, 0.6, 0.6, 0.6]
# overSSR_ret_depth10 = [0.55, 0.55, 0.56, 0.55, 0.55]
# overSSR_sys_depth10 = [0.59, 0.62, 0.64, 0.64, 0.64]
#
# overSSR_p2p_depth5 = [0.71, 0.71, 0.71, 0.71, 0.71]
# overSSR_p2p_ret_depth5 = [0.68, 0.68, 0.68, 0.68, 0.68]
# overSSR_p2p_sys_depth5 = [0.73, 0.73, 0.73, 0.73, 0.73]
#
# overSSR_p2p_depth10 = [0.75, 0.75, 0.75, 0.75, 0.75]
# overSSR_p2p_ret_depth10 = [0.73, 0.73, 0.73, 0.73, 0.73]
# overSSR_p2p_sys_depth10 = [0.77, 0.77, 0.77, 0.77, 0.77]
#
# plt.figure(figsize=(20,10))
# plt.subplot(2,1,1)
# p1, = plt.plot(hiddenLayers,hypVol_depth5,'kx-', linewidth=1, markersize=10, label='Tree Depth (TD): 5')
# p1_1, = plt.plot(hiddenLayers,hypVol_depth10,'kx--',dashes=(5, 10), linewidth=1, markersize=10, label='Tree Depth (TD): 10')
# plt.xlabel('Hidden nodes', fontsize=18)
# plt.ylabel('Hypervolume difference metric', fontsize=18)
# plt.xlim((5, 160))
# plt.xticks(hiddenLayers)
# plt.legend(handles=[p1, p1_1])
# plt.grid()
# plt.subplot(2,1,2)
# p2, = plt.plot(hiddenLayers,overSSR_depth5,'cx-', linewidth=1, markersize=10, label='Mean f1_score (TD: 5)')
# p3, = plt.plot(hiddenLayers,overSSR_ret_depth5,'bx-', linewidth=1, markersize=10, label='Ret.Sel. f1_score (TD: 5)')
# p4, = plt.plot(hiddenLayers,overSSR_sys_depth5,'rx-', linewidth=1, markersize=10, label='Sys.Sel. f1_score (TD: 5)')
# p2_2, = plt.plot(hiddenLayers,overSSR_depth10,'cx--',dashes=(5, 10), linewidth=1, markersize=10, label='Mean f1_score (TD: 10)')
# p3_3, = plt.plot(hiddenLayers,overSSR_ret_depth10,'bx--',dashes=(5, 10), linewidth=1, markersize=10, label='Ret.Sel. f1_score (TD: 10)')
# p4_4, = plt.plot(hiddenLayers,overSSR_sys_depth10,'rx--',dashes=(5, 10), linewidth=1, markersize=10, label='Sys.Sel. f1_score (TD: 10)')
# plt.xlabel('Hidden nodes', fontsize=18)
# plt.ylabel('f1_score (applied on neighborhoods)', fontsize=18)
# plt.grid()
# plt.xlim((5, 160))
# plt.xticks(hiddenLayers)
# plt.suptitle('Number of Trees: 10 (Label Encoding)')
# plt.legend(handles=[p2, p3, p4, p2_2, p3_3, p4_4])
# plt.show()
#
# plt.figure(figsize=(20,10))
# p2, = plt.plot(hiddenLayers,overSSR_depth5,'cx-', linewidth=1, markersize=10, label='Mean f1_score (n)')
# p3, = plt.plot(hiddenLayers,overSSR_ret_depth5,'bx-', linewidth=1, markersize=10, label='Ret.Sel. f1_score (n)')
# p4, = plt.plot(hiddenLayers,overSSR_sys_depth5,'rx-', linewidth=1, markersize=10, label='Sys.Sel. f1_score (n)')
# p2_2, = plt.plot(hiddenLayers,overSSR_p2p_depth5,'cx--',dashes=(5, 10), linewidth=1, markersize=10, label='Mean f1_score (p2p)')
# p3_3, = plt.plot(hiddenLayers,overSSR_p2p_ret_depth5,'bx--',dashes=(5, 10), linewidth=1, markersize=10, label='Ret.Sel. f1_score (p2p)')
# p4_4, = plt.plot(hiddenLayers,overSSR_p2p_sys_depth5,'rx--',dashes=(5, 10), linewidth=1, markersize=10, label='Sys.Sel. f1_score (p2p)')
# plt.xlabel('Hidden nodes', fontsize=18)
# plt.ylabel('f1_score', fontsize=18)
# plt.grid()
# plt.xlim((5, 160))
# plt.xticks(hiddenLayers)
# plt.title('Number of Trees: 10 || Depth of Trees: 5 || Label Encoding')
# plt.legend(handles=[p2, p3, p4, p2_2, p3_3, p4_4])
# plt.show()
#
#
#
#
#
#
#
# ############################################################# TEST validation curve
# # from sklearn.neural_network import MLPRegressor
# # from sklearn.model_selection import validation_curve
# # from sklearn.metrics import mean_squared_error, r2_score, make_scorer
# #
# # A1BS_input = pd.concat([input_catData_enc_DF,buildingInfo_norm_DF[['Ground floor area','Height']]], axis = 1)
# # A1BS_input_trainSet = A1BS_input.iloc[trainSet_ids,:]                                                                   #   - Separate input to training
# # A1BS_input_testSet = A1BS_input.iloc[testSet_ids,:]
# #
# # train_scores, valid_scores = validation_curve(MLPRegressor(activation = 'logistic', solver = 'lbfgs', alpha=0.01 ),
# #                                               A1BS_input_trainSet,A1BS_target_norm_trainSet,
# #                                               'hidden_layer_sizes',np.arange(50,450,50),
# #                                               cv = 10, scoring = make_scorer(r2_score, greater_is_better=True),
# #                                               n_jobs = -1, verbose = 2)
# #
# #
# # plt.figure(figsize=(20,10))
# # plt.plot(np.arange(50,450,50),np.mean(train_scores, axis = 1))
# # plt.plot(np.arange(50,450,50),np.mean(valid_scores, axis = 1))
# # plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # A1ΕΗ_target = pd.concat([EH_output_contData_norm_DF,output_catData_enc_DF], axis = 1)
# # A1EH3_mixed_target = pd.concat([A1ΕΗ_con_target, A1ΕΗ_cat_target], axis = 1)
# # Target = [ cost, emissions, retrofit, system, systemSize, pv, sth, heat storage, elec. storage ]
#
#
#
#
#
#
#
# #### Prediction on train set for one-hot encoding
#
# # gs_A1BS_output_label = load_GridSearchCV_object(BS_models_load_Path+finalDir[0], fileName)
# # gs_A1BS_output_oneHot = load_GridSearchCV_object(BS_models_load_Path+finalDir[1], fileName)
# # gs_A1BS_output_Binary = load_GridSearchCV_object(BS_models_load_Path+finalDir[2], fileName)
# #
# # A1BS_norm_output_oneHot_trainSet = gs.best_estimator_.predict(A1BS_input_trainSet)
# # A1BS_norm_output_oneHot_trainSet_DF = pd.DataFrame(A1BS_norm_output_oneHot_trainSet, index = A1BS_target_trainSet.index, columns = A1BS_target_trainSet.columns)
# #
# # A1BS_output_oneHot_trainSet = profiles_norm_info.inverse_transform(A1BS_norm_output_oneHot_trainSet_DF)
# # A1BS_output_trainSet_DF = pd.DataFrame(A1BS_output_oneHot_trainSet,index = A1BS_target_trainSet.index, columns = A1BS_target_trainSet.columns)
