## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         Import Libraries and functions                                              #
## ------------------------------------------------------------------------------------------------------------------- #
## =====================================================================================================================

import matplotlib.pyplot as plt
import numpy as np
import itertools as it

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         Plot grid search results function                                           #
## ------------------------------------------------------------------------------------------------------------------- #
##        Description: This function creates a plot based on grid search cross validation results.                     #
##                     Used to compare performance of different methods - variations. It also support multi-           #
##                     criteria evaluation. The user has to provide the number of subplots he wants                    #
##                     and this must be aligned to the number of evaluation metrics (n) he used for the                #
##                     grid search. Moreover the user has to specify the ploting data and specifications               #
##                     in the following dictionary format:                                                             #
##                     plotData_dict = {'plot_legend': list of variations,                                             #
##                                      'plot_coord': {'y_1': pd DF with y axis data of all variations of metric 1     #
##                                                     'x_1': list of x axis data,                                     #
##                                                      ...                                                            #
##                                                     'y_n': pd DF with y axis data of all variations of metric n     #
##                                                     'x_n': list of x axis data                                      #
##                                      'plot_labels': {'xlabel_1': x axis label of metric 1                           #
##                                                      'ylabel_1': y axis label of metric 1                           #
##                                                      ...                                                            #
##                                                      'xlabel_n': x axis label of metric n                           #
##                                                      'ylabel_n': y axis label of metric n                           #
##                                      'plot_lineSpecs': {'colorShapeMarker': List of col,shape and marker for each   #
##                                                                             variation e.g. ['bx-']                  #
##                                                                 'lineWidth': List of linewidths for each variation  #
##                                                                 'markerSize': List of marker size for each variation#
##                                      'plot_ticks': {'xticks': List of ticks for x axis}}                            #
## =====================================================================================================================

def plotGridSearchComp(subplotsNumber,plotData_dict):

    lineObject = []

    plt.figure(figsize=(20, 10))
    for subplot in np.arange(1,subplotsNumber+1):
        plt.subplot(subplotsNumber, 1, subplot)
        for variants in np.arange(1,len(plotData_dict['plot_legend'])+1):
            lineObject.append(plt.plot(plotData_dict['plot_coord']['x_'+str(subplot)],                                  # Plot results for each variation
                               plotData_dict['plot_coord']['y_'+str(subplot)].iloc[:,variants-1],
                               plotData_dict['plot_lineSpecs']['colorShapeMarker'][variants-1],
                               linewidth = plotData_dict['plot_lineSpecs']['lineWidth'][variants-1],
                               markersize = plotData_dict['plot_lineSpecs']['markerSize'][variants-1],
                               label = plotData_dict['plot_legend'][variants-1]))
        plt.xlim((plotData_dict['plot_ticks']['xticks'][0], plotData_dict['plot_ticks']['xticks'][-1]+5))
        plt.ylabel(plotData_dict['plot_labels']['ylabel_'+str(subplot)], fontsize=18)
        plt.xticks(plotData_dict['plot_ticks']['xticks'])
        plt.grid()
    plt.legend(handles = [obj[0] for obj in it.islice(lineObject,0,len(plotData_dict['plot_legend']))])
    plt.xlabel(plotData_dict['plot_labels']['xlabel_' + str(subplot)], fontsize=18)                                     # Assumes that for both metrics the same x axis is used
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 18}
    plt.rc('font', **font)
    plt.show()





# Grid search results as table

# my_dict = {'C1':[1,2,3],'C2':[5,6,7],'C3':[9,10,11]}
# for row in zip(*([key] + (value) for key, value in sorted(my_dict.items()))):
#     print(*row)

# print "{:<8} {:<15} {:<10}".format('Key','Label','Number')
# for k, v in d.iteritems():
#     label, num = v
#     print "{:<8} {:<15} {:<10}".format(k, label, num)

# dic={'Tim':3, 'Kate':2}
# print('Name Age')
# for name, age in dic.items():
#     print('{} {}'.format(name, age))