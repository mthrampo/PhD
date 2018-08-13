## Functions to treat buildings' construction age

# Import necessary libraries
import pandas as pd
import numpy as np

samples = 12802;


def assignConstructionAgeClass(exactBuildingAge):
    ## ------------------------------------------------------------------------------------------------------------------- #
    ## This function assigns buildings' to construction age classes as those given from the GIS data
    #
    # Input:    exactBuildingAge  -> Exact construction ages
    # Output:   conAgeClass       -> Assigned construction age classes

    #  [ <= 1918 ] -> Age class 1
    #  [1919 1948] -> Age class 2
    #  [1949 1978] -> Age class 3
    #  [1979 1994] -> Age class 4
    #  [1995 2001] -> Age class 5
    #  [2002 2006] -> Age class 6
    #  [2007 2009] -> Age class 7
    #  [ > 2010  ] -> Age class 8

    conAgeClass = np.zeros(samples,dtype = 'i')

    count = 0;
    for (index,sampleAge) in exactBuildingAge.iterrows():
        if int(sampleAge['Age']) <= 1918:
            conAgeClass[count] = 1;
        elif 1949 <= int(sampleAge['Age']) <= 1978:
           conAgeClass[count] = 3;
        elif 1979 <= int(sampleAge['Age']) <= 1994:
           conAgeClass[count] = 4;
        elif 1995 <= int(sampleAge['Age']) <= 2001:
           conAgeClass[count] = 5;
        elif 2002 <= int(sampleAge['Age']) <= 2006:
           conAgeClass[count] = 6;
        elif 2007 <= int(sampleAge['Age']) <= 2009:
           conAgeClass[count] = 7;
        else:
           conAgeClass[count] = 8;
        count += 1

    return conAgeClass
