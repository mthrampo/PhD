## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         Import Libraries and functions                                              #
## ------------------------------------------------------------------------------------------------------------------- #
## =====================================================================================================================

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier
import winsound
import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda
from keras.layers import Dropout
from keras.callbacks import History, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from keras import regularizers

import livelossplot

from keras.callbacks import TensorBoard

from paretoFrontComparisonMetrics import hyperVolumeIndex_loss_func, overallSuccessfulSelectionRate_score,\
    coeffOfDetermination,hyperVolumeIndex_keras_loss_func

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         MLPerceptron function                                                       #
## ------------------------------------------------------------------------------------------------------------------- #
##              Description: This function creates and trains a multi-layer Perceptron Neural Network                  #
##                           based on given input and target and returns the results. It works for                     #
##                           classification, regression or mixed problems by setting the MLP_type parameter.           #
##                           For all type of problems it supports the following functionalities:                       #
##                           - Simple training: Simple training with cross validation given specific parameters.       #
##                                              It returns the final score based on scoringMethod.                     #
##                           - Grid search: Grid search cross validation to optimize based on given scoringMethod.     #
##                                          It returns the grid search object.                                         #
##                           OptimizeParameters is used to select between those two functionalities.                   #
## =====================================================================================================================


def MLPerceptron(MLP_input,MLP_target,MLP_type,optimizeParameters, scoringMethod):

    if scoringMethod == 'mse':                                                                                          # Create scorer based on given scoring method
        selScorer = {'mse': make_scorer(mean_squared_error, greater_is_better=False)}                                   # and define if we need minimization or optimization
        multiScoring = False
        # selScorer = mean_squared_error
        # maximize_boolean = False
    elif scoringMethod == 'r2':
        selScorer = {'r2': make_scorer(r2_score, greater_is_better=True)}
        multiScoring = False
        # selScorer = r2_score
        # maximize_boolean = True
    elif scoringMethod == 'mse,r2':
        multiScoring = True
        selScorer = {'mse': make_scorer(mean_squared_error, greater_is_better = False),
                     'r2': make_scorer(r2_score, greater_is_better = True)}
    elif  scoringMethod == 'hypervolume':
        # selScorer = hyperVolumeIndex_loss_func
        selScorer =  make_scorer(hyperVolumeIndex_loss_func, greater_is_better=False)
        multiScoring = False
        # maximize_boolean = False

    folds_i = int(input("  -- Select number of cross validation folds [e.g. 5]: "))                                     # Ask user for the number of folds for cross validation
        # maxIter = 10

    if MLP_type == 'Regression':                                                                                        # Regression Problems (Predict continuous outputs)

        if optimizeParameters == 'True':                                                                                # Optimize parameters via grid search cross validation

                                                                                                                        # Ask user for to provide parameters to be optimized
            activationFs_i = input("  -- Select activation functions [e.g. logistic,tanh ]: ").split(',')               #   - Activation function for hidden layer(s)
            solvers_i = input("  -- Select solvers [e.g. lbfgs ]: ").split(',')                                         #   - Optimization solver
            hidNodes_i = [int(hidNod) for hidNod in (input("  -- Select number of hidden node pairs"                    #   - Hidden nodes per hidden layer(s)
                                                           " [e.g. 10, 20, (10,20) ]: ").split(','))]
            learningRateEv_i = input("  -- Select learning rate exvolution methods "                                    #   - Learning rate evolution (when SGD used as solver)
                                     "[e.g. invscaling,constant ]: ").split(',')
            alphas_i = [float(a) for a in (input("  -- Select alphas [e.g. 0.1,0,1,10 ]: ").split(','))]                #   - Regularization parameter

            parameters = {'activation': activationFs_i,                                                                 # Parameter grid
                          'solver': solvers_i,
                          'hidden_layer_sizes': hidNodes_i,
                          'learning_rate': learningRateEv_i,
                          'alpha':alphas_i}

            reg_MLP = MLPRegressor(verbose=10)                                                                                    # Create a multi-layer Perceptron regressor object
            # gs_scorer = make_scorer(selScorer, greater_is_better = maximize_boolean)

            if multiScoring:                                                                                            # For multiscoring case refit with given score the estimator
                refit_i = 'mse'                                                                                         # using the best found parameters on the whole dataset.
            else:
                refit_i = True

            gs = GridSearchCV(estimator = reg_MLP,                                                                      # Create a grid search object
                              param_grid = parameters,
                              # scoring = gs_scorer,
                              scoring = selScorer,
                              cv = folds_i,
                              verbose = 3,
                              n_jobs = 1,
                              return_train_score = True,
                              refit = refit_i)

            gs.fit(MLP_input,MLP_target)                                                                                # Perform grid search

            bestParameters = gs.best_params_                                                                            # Retrieve best parameters

            print('#### Optimization results ####')                                                                     # Print best parameters
            print('Best estimator has the folowing parameters: ')
            print(bestParameters)
            print('##############################')

            return gs                                                                                                   # Return Grid search object

        else:
            hidNodes_i = int(input("  -- Select number of hidden nodes [e.g. 10]: "))
            activationF_i = input("  -- Select activation function [e.g. 'logistic']: ")
            solver_i = input("  -- Select solver [e.g. 'lbfgs']: ")
            maxIter_i = int(input("  -- Select number of iterations [e.g. 8]: "))
            # for hidLayers in range(10, 40, 10):
            mse = 0.0
            coeffOfDet = 0.0
            for iter in range(1,maxIter_i):
                reg_MLP = MLPRegressor(hidden_layer_sizes = (hidNodes_i), activation = activationF_i, solver = solver_i)

                mse_val = 0.0
                coeffOfDet_val = 0.0

                kf = KFold(n_splits = folds_i)
                for train_indices, val_indices in kf.split(MLP_input):
                    x_train_set = MLP_input.reindex(train_indices+1)
                    y_train_set = MLP_target.reindex(train_indices+1)

                    x_valid_set = MLP_input.reindex(val_indices+1)
                    y_valid_set = MLP_target.reindex(val_indices+1)

                    reg_MLP.fit(x_train_set, y_train_set)

                    y_pred = reg_MLP.predict(x_valid_set)

                    coeffOfDet_val += r2_score(y_valid_set, y_pred)
                    mse_val += mean_squared_error(y_valid_set, y_pred)**0.5
                coeffOfDet += (coeffOfDet_val / folds_i)
                mse += (mse_val / folds_i)
            mse_f = mse/(maxIter_i-1)
            coeffOfDet_f = coeffOfDet/(maxIter_i-1)

            print(hidNodes_i,': %.4f' % mse_f, ' | %.4f' %coeffOfDet_f)

            return mse_f

def hybridMLAlgorithm(ML_input,ML_con_target,ML_cat_target,optimizeParameters):

    folds_i = int(input("  -- Select number of cross validation folds [e.g. 5]: "))                                     # Ask user for the number of folds for cross validation

    # from sklearn.utils import class_weight                                                                              # Handle class imbalance
    #
    # weights = class_weight.compute_class_weight('balanced', list(range(0, 5)), Y_train_)
    # weights = dict(enumerate(weights))
    # print(weights)
    #
    # A1ΕΗ2_target_norm_trainSet['ret.Sel._PFP_1'].value_counts()

    if optimizeParameters == 'True':
        print('Not implemented yet')

    elif optimizeParameters == 'False':

        # Ask from user to give NN parameters
        maxIter_i = int(input("  -- Select number of iterations [e.g. 8]: "))
        print ('* Input parameters for NN regressor *')
        hidNodes_i = int(input("  -- Select number of hidden nodes [e.g. 10]: "))
        activationF_i = input("  -- Select activation function [e.g. 'logistic']: ")
        solver_i = input("  -- Select solver [e.g. 'lbfgs','adam']: ")
        print('* Input parameters for random forest classifier *')
        trees_i = int(input("  -- Select number of trees [e.g. 10 ]: "))
        depth_i = int(input("  -- Select depth for trees [e.g. 5 ]: "))

        # maxIter_i = 10

        hp_index = 0.0
        ossr_ret_n = 0.0
        ossr_sys_n = 0.0
        ossr_ret = 0.0
        ossr_sys = 0.0

        for iter in range(1,maxIter_i):

            # Create one regression and one classification NN objects with provided parameters
            # reg_MLP = MLPRegressor(hidden_layer_sizes = (hidNodes_i), activation = activationF_i, solver = solver_i)
            reg_MLP = create_keras_model(inputShape = ML_input.shape[1], outputShape = ML_con_target.shape[1],
                                         optimizer = solver_i, activation = activationF_i, hidden_layers = 4,
                                         neurons= hidNodes_i)
            classifier = RandomForestClassifier(max_depth = depth_i, n_estimators= trees_i, random_state = 0, oob_score = True)

            hp_index_val = 0.0
            ossr_ret_val = 0.0
            ossr_sys_val = 0.0
            ossr_ret_n_val = 0.0
            ossr_sys_n_val = 0.0

            kf = KFold(n_splits=folds_i)
            for train_indices, val_indices in kf.split(ML_input):

                # Divide data set into training and validation set
                x_train_set = ML_input.iloc[train_indices,:]
                y_cat_train_set = ML_cat_target.iloc[train_indices,:]
                y_con_train_set = ML_con_target.iloc[train_indices,:]

                x_valid_set = ML_input.iloc[val_indices,:]
                y_cat_valid_set = ML_cat_target.iloc[val_indices,:]
                y_con_valid_set = ML_con_target.iloc[val_indices,:]

                # # Separate also between continuous and categorical data
                # y_train_con_set = y_train_set.iloc[:,:20]
                # y_train_cat_set = y_train_set.iloc[:,20:]
                #
                # y_valid_con_set = y_valid_set.iloc[:,:20]
                # y_valid_cat_set = y_valid_set.iloc[:,20:]

                # Train ML algorithms
                reg_MLP.fit(x_train_set, y_con_train_set)
                classifier.fit(x_train_set, y_cat_train_set)

                # Prediction on validation data
                y_pred_con_set = reg_MLP.predict(x_valid_set)
                y_pred_cat_set = classifier.predict(x_valid_set)

                # Calculate goodness of fit
                hp_index_val += hyperVolumeIndex_loss_func(y_con_valid_set, y_pred_con_set)
                (ret_score_n,sys_score_n,ret_score,sys_score) = overallSuccessfulSelectionRate_score(y_con_valid_set, y_pred_con_set, y_cat_valid_set, y_pred_cat_set)
                ossr_ret_n_val += ret_score_n
                ossr_sys_n_val += sys_score_n
                ossr_ret_val += ret_score
                ossr_sys_val += sys_score

            hp_index += (hp_index_val / folds_i)
            ossr_ret_n += (ossr_ret_n_val / folds_i)
            ossr_sys_n += (ossr_sys_n_val / folds_i)
            ossr_ret += (ossr_ret_val / folds_i)
            ossr_sys += (ossr_sys_val / folds_i)

        hp_index_f = hp_index / (maxIter_i - 1)
        ossr_ret_n_f = ossr_ret_n / (maxIter_i - 1)
        ossr_sys_n_f = ossr_sys_n / (maxIter_i - 1)
        ossr_ret_f = ossr_ret / (maxIter_i - 1)
        ossr_sys_f = ossr_sys / (maxIter_i - 1)
        ossr_f = (ossr_ret_f + ossr_sys_f)/2
        ossr_n_f = (ossr_ret_n_f + ossr_sys_n_f) / 2


        duration = 1000  # millisecond                                                                                  # Inform user with sound that the sumulation has finished
        freq = 440  # Hz
        for i in range(1,5):
            winsound.Beep(freq, duration)
            time.sleep(1)

        print(hidNodes_i,depth_i, ': %.4f' % hp_index, ' | %.2f' % ossr_n_f, ' | %.2f' % ossr_ret_n_f, ' | %.2f' % ossr_sys_n_f, ' |||| %.2f' % ossr_f, ' | %.2f' % ossr_ret_f, ' | %.2f' % ossr_sys_f)

        return classifier

# def kerasNN(NN_input,NN_target,optimizeParameters = None, scoringMethod = None):                                      # TRY interactive plots (does not work - only in jupyter)
#     plot_losses = livelossplot.PlotLossesKeras(figsize=(8, 4))
#
#     # hidNodes_i = int(input("  -- Select number of hidden nodes [e.g. 10]: "))
#     # activationF_i = input("  -- Select activation function [e.g. 'logistic']: ")
#     # solver_i = input("  -- Select solver [e.g. 'lbfgs']: ")
#     # maxIter_i = int(input("  -- Select number of iterations [e.g. 8]: "))
#     maxIter_i = 1
#     folds_i = 10
#
#     for iter in range(1, maxIter_i):
#         nn = create_keras_model(NN_input.shape[1], NN_target.shape[1])
#
#         kf = KFold(n_splits=folds_i)
#         for train_indices, val_indices in kf.split(NN_input):
#             x_train_set = NN_input.reindex(train_indices + 1)
#             y_train_set = NN_target.reindex(train_indices + 1)
#
#             x_valid_set = NN_input.reindex(val_indices + 1)
#             y_valid_set = NN_target.reindex(val_indices + 1)
#
#             nn.fit(x_train_set, y_train_set,
#                    epochs=10, validation_data=(x_valid_set, y_valid_set),
#                    callbacks=[plot_losses],verbose=0)

    # history = History()  # keep logs in this object

    # np.random.seed(4)

def kerasNN(NN_input,NN_target,optimizeParameters, randomGenSeed, checkpointPath = True, scoringMethod = None, NN_params = None):

    x_train, x_val, y_train, y_val = train_test_split(NN_input, NN_target, train_size = 0.85, random_state = randomGenSeed)

    if optimizeParameters == True:
        # filepath = checkpointPath+ 'weights-improvement-{epoch:02d}-{val_mean_squared_error:.6f}.hdf5'                      # Add checkpoints
        # checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')
        history = History()                                                                                               # Print epoch performance
        callBachTF = TensorBoard(log_dir='./Graph', histogram_freq=0,write_graph=True, write_images=True)                 # For tensorboard

        print('In2')

        kfold = 10

        # classifier = KerasRegressor(build_fn = create_keras_model(NN_input.shape[1],NN_target.shape[1]), epochs=30, batch_size=1000, verbose=4)

        regressor = KerasRegressor(build_fn = create_keras_model, epochs = 100, batch_size=1000, verbose=1)

        neurons = np.arange(10,101,10) #[5,10,15,20,25,30]  # [50, 100, 150, 200, 250, 300]
        hidden_layers = [1]  # [1,2,3,4,5,6]
        # optimizer = ['sgd']  # ['adam','sgd']
        activation = ['relu']  # ,'softmax','sigmoid','tanh']
        # dropout = [0.1]  # [0.1,0.5,1,2]

        parameters = dict(neurons = neurons,
                          hidden_layers = hidden_layers,
                          # optimizer = optimizer,
                          activation = activation,
                          # dropout = dropout
                          )

        # selScorer = {'mse': make_scorer(mean_squared_error, greater_is_better=False),
        #              'r2': make_scorer(r2_score, greater_is_better=True)}

        selScorer = 'mean_squared_error'

        gs = GridSearchCV(estimator = regressor,
                          param_grid = parameters,
                          scoring=selScorer,
                          n_jobs = 1,
                          cv = kfold,
                          verbose = 3,
                          return_train_score=True) #,
                          # refit = 'mse')

        gs_start = time.perf_counter()

        grid_result = gs.fit(NN_input,
                             NN_target,
                             callbacks=[history],
                             verbose=1,
                             validation_data=(x_val, y_val))
        # grid_result = gs.fit(NN_input, NN_target, callbacks=[checkpoint], verbose=0, validation_split=0.15)

        gs_end = time.perf_counter()

        print("Grid search CV summary")
        print("======================")
        print("Time: %.2f s" % (gs_end - gs_start))
        print("Best: %f using %s" % (abs(grid_result.best_score_), grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (abs(mean), stdev, param))

        # print("Best: %f using %s" % (abs(grid_result.best_score_), grid_result.best_params_))                               # summarize results (sinlge criteria)
        # mse_means = grid_result.cv_results_['mean_test_mse']
        # mse_stds = grid_result.cv_results_['std_test_mse']
        # r2_means = grid_result.cv_results_['mean_test_r2']
        # r2_stds = grid_result.cv_results_['std_test_r2']
        # params = grid_result.cv_results_['params']
        # for mean_1, stdev_1,mean_2, stdev_2, param in zip(mse_means, mse_stds, r2_means, r2_stds, params):
        #     print("%f (%f) || %f (%f) with: %r" % (abs(mean_1), stdev_1,abs(mean_2),stdev_2, param))

        return gs, grid_result

    elif optimizeParameters == False:

        keras_NN = create_keras_model(neurons =  NN_params['dense_layers'],
                                activation = NN_params['activation'],
                                optimizer = NN_params['optimizer'])

        history = keras_NN.fit(x_train,
                            y_train,
                            epochs = NN_params['epochs'],
                            batch_size = NN_params['batch_size'],
                            verbose=0,
                            validation_data=(x_val, y_val))

        keras_NN.summary()




# def create_keras_model(inputShape = 6, outputShape = 28, optimizer='adam', activation='sigmoid', hidden_layers=1, neurons=5):     # For BS label encoding


# def create_keras_model(inputShape=27, outputShape=28, optimizer='adam', activation='sigmoid', hidden_layers=1, neurons=5): # For BS one-hot encoding
# def create_keras_model(inputShape=36, outputShape=28, optimizer='adam', activation='sigmoid', hidden_layers=1, neurons=5): # For BS keras one-hot encoding

def create_keras_model(inputShape = 36, outputShape = 20, optimizer='adam', activation='sigmoid', hidden_layers=1, neurons=5):  # For EH cost&emissions one-hot encoding
# def create_keras_model(inputShape= 6, outputShape= 20, optimizer='adam', activation='sigmoid', hidden_layers=1, neurons=5):
    model = Sequential()                                                                                                # Initialize the constructor

    model.add(Dense(neurons, activation=activation, input_shape=(inputShape,)))                                         # Add an input layer
    # model.add(Dropout(dropout))

    for i in range(hidden_layers):                                                                                      # Add as many hidden layers as given in the input
        model.add(Dense(neurons, activation=activation, kernel_regularizer = regularizers.l2(0.01)))
        # model.add(Dropout(dropout))

    model.add(Dense(outputShape, activation='linear',kernel_regularizer = regularizers.l2(0.01)))                       # Add an output layer

    # model.compile(loss='mse', optimizer=optimizer, metrics=['mse',coeffOfDetermination])                              # compile model
    # model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])                                                     # mse

    # model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse'])

    # model.add(Lambda(lambda x: x^2))#hyperVolumeIndex_loss_func(x[0],x[1])))

    model.compile(loss = 'mse', optimizer=optimizer, metrics = ['mse'])

    return model



# def create_keras_model(inputShape= 6, outputShape= 20, optimizer='adam', activation='sigmoid', hidden_layers=1, neurons=5):
#
#     from keras.models import Model
#     from keras.layers import Input
#     from keras.layers import Dense
#
#
#     input_layer = Input(shape = (inputShape,))
#
#     hidden_layer_1 = Dense(neurons, activation='relu')(input_layer)
#     hidden_layer_2 = Dense(neurons, activation='relu')(hidden_layer_1)
#     output = Dense(outputShape, activation='linear')(hidden_layer_2)
#
#     # custom_loss = Lambda(lambda x: x^2)(output)
#     #
#     model = Model(inputs = input_layer, outputs = output)
#
#     model.compile(loss = hyperVolumeIndex_loss_func, optimizer=optimizer, metrics=['mse'])
#
#     return model



