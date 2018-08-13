## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         Import Libraries and functions                                              #
## ------------------------------------------------------------------------------------------------------------------- #
## =====================================================================================================================

from pygmo import hypervolume
import numpy as np
import math as mt
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix,f1_score

# from main_ExecutionFile import get_selection_coordinates

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                        hyperVolumeIndex_loss function                                               #
## ------------------------------------------------------------------------------------------------------------------- #
##          Description: This function calculates the hypervolume difference metric introduced in                      #
##                       https://link.springer.com/chapter/10.1007/978-3-642-30976-2_61                                #
##                       More specifically it calculates the area between the target pareto front                      #
##                       and the one derived as output from a ML algorithm. This is done by specifying                 #
##                       a reference point.                                                                            #
## =====================================================================================================================

def hyperVolumeIndex_loss_func(target,output):


    refPoint = [1.5,1.5]                                                                                                # Define reference point by which the hypervolume is calculated
                                                                                                                        # Valid only for min-max normalization between [0,1].
                                                                                                                        # TO BE CHECKED FOR OTHER NORM METHODS !!
    refPF = np.array(target)
    outPF = np.array(output)

    I_H = np.empty(shape = target.shape[0], dtype=float)

    for i, (refPf_i, outPF_i) in enumerate(zip(refPF,outPF)):
        ref_hv_obj = hypervolume(points = refPf_i.reshape(10,-1,order='F'))                                             # reshape vector to array of the form:
        out_hv_obj = hypervolume(points = outPF_i.reshape(10,-1,order='F'))                                             # points = [[x1,y1],[x2,y2],...,[xn,yn]]
        ref_hv = ref_hv_obj.compute(refPoint)
        out_hv = out_hv_obj.compute(refPoint)
        I_H[i] = abs(ref_hv - out_hv)

    hpv_index = I_H.mean()

    return hpv_index

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                   Overall successful Selection Rate Function                                        #
## ------------------------------------------------------------------------------------------------------------------- #
##          Description: This function calculates the mean f1_score of retrofit and system selection                   #
##                       by defining for each pareto front a circle neighborhood. Thus the output                      #
##                       selection is compared to the target selections of all neighbors. If there                     #
##                       are no neighbors in the neighborhood then neighbor is considered the one                      #
##                       with the least euclidean distance.                                                            #
## =====================================================================================================================

def overallSuccessfulSelectionRate_score(target_cont,output_cont,target_cat,output_cat):

    output_cont = pd.DataFrame(output_cont, columns=target_cont.columns, index=target_cont.index)
    output_cat = pd.DataFrame(output_cat, columns=target_cat.columns, index=target_cat.index)

    offset = 0.15                                                                                                       # Set neighborhood radius offset (in percentage)

    neigborhood_Info = pd.DataFrame(columns = ['center_x','center_y','radius','Circle neighbours',                      # Define neighborhood information dataframe
                                               'cls. dist. neighbours','all neighbors'],
                                    index = np.arange(1,target_cont.shape[0]*10+1))

    count = 0
    for target_sample,output_sample in zip(target_cont.iterrows(),output_cont.iterrows()):

        # target_sample[0] = output_sample[0] = index
        # target_sample[1][0:10] --> Contains the cost values
        # target_sample[1][10:20] --> Contains the emission values

        index = count

        x_range = max(target_sample[1][0:10]) - min(target_sample[1][0:10])                                             # Calculate PF cost range
        y_range = max(target_sample[1][10:20]) - min(target_sample[1][10:20])                                           # Calculate PF emissions range

        neigborhood_Info.at[index*10+1:(index+1)*10,neigborhood_Info.columns[2]] = mt.sqrt((x_range*offset)**2+         # Calculate cirlce neighborhood's radius for each PFP
                                                                                           (y_range*offset)**2)
        neigborhood_Info.loc[index*10+1:(index+1)*10,neigborhood_Info.columns[0]] = target_sample[1][0:10].values       # Set cirlce neighborhood's x coordinate for each PFP
        neigborhood_Info.loc[index*10+1:(index+1)*10, neigborhood_Info.columns[1]] = target_sample[1][10:20].values     # Set cirlce neighborhood's y coordinate for each PFP

        selectedRange = neigborhood_Info.loc[index*10+1:(index+1)*10,:].copy()

        for i, neighborhood in selectedRange.iterrows():
            isthereNeighbor = [((output_sample[1][0:10]-neighborhood['center_x'])**2).values +                          # Check if there are any neighbours within circle neighborhood
                               ((output_sample[1][10:20] - neighborhood['center_y']) ** 2).values
                               <= neighborhood['radius']**2]
            selectedRange.at[i, 'Circle neighbours'] = [i for i,s in enumerate(isthereNeighbor) if 'True' in s]         # Store potential neighbours

        minEuclDist = cdist(output_sample[1].values.reshape(2,-1).transpose(),                                                 # calculate all PFP euclidean distances and
                                    target_sample[1].values.reshape(2,-1).transpose())
        for it, pfp_output_dist in enumerate(minEuclDist):
            selectedRange.at[index*10+it+1, 'cls. dist. neighbours'] = np.where(pfp_output_dist ==
                                                                                np.min(pfp_output_dist))                # select the one with the least euclidean distance
            if not selectedRange.at[index*10+it+1, 'Circle neighbours']:                                                # Store all neighbours
                selectedRange.at[index*10+it+1, 'all neighbors'] = selectedRange.at[index*10+it+1, 'cls. dist. neighbours']
            else:
                selectedRange.at[index*10+it+1, 'all neighbors'] = selectedRange.at[index*10+it+1,'Circle neighbours']

        neigborhood_Info.loc[index * 10 + 1:(index + 1) * 10, :] = selectedRange

        count += 1

    target_cat_Updated = pd.DataFrame(index = np.arange(1,int(neigborhood_Info.shape[0]/10)+1), columns = target_cat.columns)

    count = 0
    for index, target_cat_sample in enumerate(target_cat.iterrows()): #, output_cat_sample in zip(target_cat.iterrows(), output_cat.iterrows()):

        for i,neighbor in enumerate(neigborhood_Info.loc[index * 10 + 1:(index + 1) * 10, 'all neighbors']):            # Update target based on defined unique neighbors
            target_cat_Updated.at[count+1,target_cat.columns[i]] = \
                np.unique(list(target_cat_sample[1][list(int(i) for i in neighbor[0])].values))[0]
            target_cat_Updated.at[count+1, target_cat.columns[i + 10]] = \
                np.unique(list(target_cat_sample[1][[x+10 for x in list(int(i) for i in neighbor[0])]].values))[0]
        count += 1

    sum_f1_score_ret_n = 0.0
    sum_f1_score_sys_n = 0.0
    count = 0
    for target_sample,output_sample in zip(target_cat_Updated.iterrows(), output_cat.iterrows()):                       # Calculate f1 score for each building retrofit and system selection
        sum_f1_score_ret_n += f1_score(list(target_sample[1][0:10]),list(output_sample[1][0:10]),average='micro')       # based on neighbours
        sum_f1_score_sys_n += f1_score(list(target_sample[1][10:20]), list(output_sample[1][10:20]), average='micro')
        count += 1

    f1_score_ret_n = sum_f1_score_ret_n/count                                                                               # Calculate mean f1 score for all retrofit selections
    f1_score_sys_n = sum_f1_score_sys_n/count                                                                               # Calculate mean f1 score for all system selections

    sum_f1_score_ret = 0.0
    sum_f1_score_sys = 0.0
    count = 0
    for target_sample,output_sample in zip(target_cat.iterrows(), output_cat.iterrows()):                               # Calculate f1 score for each building retrofit and system selection
        sum_f1_score_ret += f1_score(list(target_sample[1][0:10]),list(output_sample[1][0:10]),average='micro')         # point by point
        sum_f1_score_sys += f1_score(list(target_sample[1][10:20]), list(output_sample[1][10:20]), average='micro')
        count += 1

    f1_score_ret = sum_f1_score_ret/count                                                                               # Calculate mean f1 score for all retrofit selections
    f1_score_sys = sum_f1_score_sys/count                                                                               # Calculate mean f1 score for all system selections


    # mean_f1_score = (f1_score_ret + f1_score_sys)/2                                                                     # Average f1 score as the some of scores of retrofit and system

    return f1_score_ret_n,f1_score_sys_n,f1_score_ret,f1_score_sys


## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                             Coefficient of determination (R^2) metric (keras)                                       #
## ------------------------------------------------------------------------------------------------------------------- #
##          Description: This metric calculates the coefficient of determination. It is implemented for                #
##                       for keras NNs that do not have it as build-in metric.                                         #
## =====================================================================================================================

def coeffOfDetermination(y_true, y_pred):

    from keras import backend as K

    SS_res =  K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    r2 = (1 - SS_res/(SS_tot + K.epsilon()))                                                                            # used to avoid devision by zero

    return r2


## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                             Coefficient of determination (R^2) metric (keras)                                       #
## ------------------------------------------------------------------------------------------------------------------- #
##          Description: This metric calculates the coefficient of determination. It is implemented for                #
##                       for keras NNs that do not have it as build-in metric.                                         #
## =====================================================================================================================

def hyperVolumeIndex_keras_loss_func(y_true, y_pred):

    from keras import backend as K
    import tensorflow as tf

    y_true_np = K.eval(y_true)
    y_pred_np = K.eval(y_pred)

    ref_point = [1.0, 1.0]
    ref_PF = np.array(y_true)  # ref_PF = K.variable(y_true)
    out_PF = np.array(y_pred)  # out_PF = K.variable(y_pred)



    hv_d = np.empty(shape=y_true.shape[0], dtype=float)

    for i, (ref_point_i, out_PF_i) in enumerate(zip(ref_point, out_PF)):
        ref_hv_obj = hypervolume(points=ref_point.reshape(10, -1, order='F'))
        out_hv_obj = hypervolume(points=ref_point.reshape(10, -1, order='F'))
        ref_hv = ref_hv_obj.compute(ref_point)
        out_hv = out_hv_obj.compute(ref_point)
        hv_d[i] = abs(ref_hv - out_hv)

    hv_d_tensor = tf.convert_to_tensor(hv_d)

    return hv_d


    # # return -10. * (K.mean(K.square(y_pred - y_true)))
    #
    #
    # # y_true_c = K.cast(y_true, dtype='float32')
    # # y_pred_c = K.cast(y_pred, dtype='float32')
    #
    # hpv_index = K.placeholder(dtype = 'float32')
    # hv = tf.placeholder(dtype = 'float32')
    # hv_k_loss = K.function([y_true, y_pred], [hv], [hyperVolumeIndex_loss_func(y_true, y_pred)])
    # hv_k = hv_k_loss([y_true, y_pred])
    # # hv_k = tf.py_func(hyperVolumeIndex_loss_func, [hpv_index] ,tf.float32)

    return hv_k