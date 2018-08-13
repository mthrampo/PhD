## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         Import Libraries and functions                                              #
## ------------------------------------------------------------------------------------------------------------------- #
## =====================================================================================================================

from scipy.io import loadmat
import pandas as pd
import numpy as np

## =====================================================================================================================
## ------------------------------------------------------------------------------------------------------------------- #
##                                         loadBuildingInfo function                                                   #
## ------------------------------------------------------------------------------------------------------------------- #
##              Description: This function loads building information from .mat files. Those files contain             #
##                           building information, building simulation results and building system and                 #
##                           envelope optimization results. More specifically the information retrieved                #
##                           from this function is:                                                                    #
##                           - Building Data (derived from ArcGIS)                                                     #
##                           - Energy demand Data (derived from CESAR)                                                 #
##                           - Energy Hub Data (derived from AIMMS)                                                    #
##                           The information derived is stored in dataframes.                                          #
## =====================================================================================================================


def loadBuildingInfo(path):

    samples = 12802;

    # Building data
    bd_fileName = ['buildingAge.mat', 'buildingArea.mat', 'buildingEnCarriers.mat', 'buildingEnCarriers.mat',
                   'buildingGFarea.mat', 'buildingHeight.mat', 'buildingID.mat', 'buildingRarea.mat',
                   'buildingRorient.mat', 'buildingRslope.mat', 'BuildingType.mat']
    bd_dictName = ['bcon', 'buildingArea', 'benc', 'benc', 'bgfarea', 'bheig', 'bid', 'buildingRarea', 'brorient',
                   'brslop', 'btype']

    bd_numOfFiles = len(bd_fileName)
    bd_input = np.zeros((bd_numOfFiles,samples), dtype='f') #array([[0] * samples for i in range(bd_numOfFiles)], dtype='f')


    for file_id in range(bd_numOfFiles):
        loadFile = loadmat(path + bd_fileName[file_id])
        for sample in range(samples):
            if file_id != 1:
                bd_input[file_id][sample] = loadFile.get(bd_dictName[file_id])[sample][0][0][0]
            elif file_id == 1:
                bd_input[file_id][sample] = loadFile.get(bd_dictName[file_id])[sample][0]

    # Energy demand data

    ed_fileName = ['EH_annual_h.mat', 'EH_annual_e.mat', 'EH_peak_h.mat', 'EH_peak_e.mat']
    ed_dictName = ['EH_annual_h_f', 'EH_annual_e_f', 'EH_peak_h_f', 'EH_peak_e_f']

    ed_numOfFiles = len(ed_fileName)
    ed_retOptions = 7;

    ed_input = np.zeros((ed_numOfFiles,ed_retOptions,samples), dtype='f')

    for file_id in range(ed_numOfFiles):
        loadFile = loadmat(path + ed_fileName[file_id])
        for ret_opt in range(ed_retOptions):
            for sample in range(samples):
                ed_input[file_id][ret_opt][sample] = loadFile.get(ed_dictName[file_id])[sample][ret_opt]

    # Energy hub output

    paretoPoints_max = 10
    energyHub_outCol = 25

    energyHub_out = np.zeros((samples, paretoPoints_max, energyHub_outCol), dtype='f')

    loadFile = loadmat(path + 'AIMMSOut_12802_23022018.mat')

    for sample in range(samples):
        for paretoPoint in range(paretoPoints_max):
            for col in range(energyHub_outCol):
                energyHub_out[sample][paretoPoint][col] = loadFile.get('AIMMSOut')[sample][0][paretoPoint][col]


    # Building data database

    bd_labels = ['Age','Total Area','Heating en.Carrier','DHW en.Carrier','Ground floor area','Height','ID','Roof area','Roof orientation','Roof slope','Building type']
    buildingInfo = pd.DataFrame(bd_input.transpose(),index = np.arange(1,samples+1),columns = bd_labels)

    # Energy demand databases

    ret_labels = ['Base case','Full ret.','Wall ret.','Window ret.','Wall & Window ret.','Roof ret.','Wall & Win & Roof ret.']
    enDemand_annual_h = pd.DataFrame(ed_input[0].transpose(),index = np.arange(1,samples+1), columns = ['AH_' + str for str in ret_labels])
    enDemand_annual_e = pd.DataFrame(ed_input[1].transpose(), index=np.arange(1, samples + 1), columns= ['AE_' + str for str in ret_labels])
    enDemand_peak_h = pd.DataFrame(ed_input[2].transpose(), index=np.arange(1, samples + 1), columns= ['PH_' + str for str in ret_labels])
    enDemand_peak_e = pd.DataFrame(ed_input[3].transpose(), index=np.arange(1, samples + 1), columns= ['PE_' + str for str in ret_labels])

    # Energy Hub output database

    eho_labels = ['Cost','Emissions','Base case','Full retrofit','Wall retrofit','Window retrofit','Wall & Window retrofit','Roof retrofit','Roof & Wall & Window retrofit',
                  'Oil boiler','Gas boiler','Bio boiler','CHP','ASHP','GSHP','PV','Solar thermal','Heat storage','Electricity storage','Gas boiler (existing)','Oil boiler (existing)',
                  'Bio boiler (existing)','Distict heating (existing)','Electricity (existing)','Heat pump (existing)']
    EH_output = pd.DataFrame(energyHub_out[0],index = np.arange(1,paretoPoints_max+1), columns = eho_labels)

    for sample in range(1,samples):
        temp_EH_output = pd.DataFrame(energyHub_out[sample],index = np.arange(1,paretoPoints_max+1), columns = eho_labels)
        EH_output = pd.concat([EH_output,temp_EH_output])

    # Return databases

    return (buildingInfo, enDemand_annual_h, enDemand_annual_e, enDemand_peak_h, enDemand_peak_e, EH_output)