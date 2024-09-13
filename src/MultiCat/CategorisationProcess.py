'''
Created on 23 Aug 2018

@author: Andreagiovanni Reina.
University of Sheffield, UK.
'''

import sys
import os
import configparser
import numpy.random as rand
from MultiCat.MyTypes import AllocationType
from MultiCat import Population
import json

DEBUG=True

DEFAULT_PROPERTIES_FILENAME = "/Users/joefresna/MultiCategorisation/conf/MultiCat.config"

if __name__ == '__main__':
    if DEBUG: 
        print("Process Started")
    if len(sys.argv)>1 :
        configFile = sys.argv[1]
    else:
        configFile = DEFAULT_PROPERTIES_FILENAME
    
    if not (os.path.isfile(configFile)):
        print("Impossible to open config file: ", configFile)
        sys.exit()
    
    if DEBUG:
        print("Reading properties file: " + configFile)
    # opening the config file
    config = configparser.ConfigParser()
    config.read(configFile)
   
    ### Load parameters from config file
    ## -- Experiment params
    randomSeed = config.getint('experiment', 'randomSeed')
    numberOfExperiments = config.getint('experiment', 'numberOfExperiments')
    outputTxtFile = config.get('experiment', 'outputTxtFile')
    outputPdfFile = config.get('experiment', 'outputPdfFile')
    cluster = config.getboolean('experiment', 'cluster')
    if (cluster): DEBUG=False
    ## -- Features params
    numFeatures = config.getint('Features', 'numFeatures')
    featuresAccuracies = json.loads(config.get('Features', 'featuresAccuracies'))
    allocationTypeStr = config.get('Features', 'allocationType')
    if (allocationTypeStr == 'random'):
        allocationType = AllocationType.RANDOM
    elif  (allocationTypeStr == 'split'):
        allocationType = AllocationType.SPLIT
    elif  (allocationTypeStr == 'on-noise'):
        allocationType = AllocationType.ON_NOISE
    elif  (allocationTypeStr == 'on-noise-odds'):
        allocationType = AllocationType.ON_NOISE_ODDS
    elif  (allocationTypeStr == 'on-noise-var'):
        allocationType = AllocationType.ON_NOISE_VAR
    elif  (allocationTypeStr == 'on-noise-err'):
        allocationType = AllocationType.ON_NOISE_ERR
    elif  (allocationTypeStr == 'on-noise-llr'):
        allocationType = AllocationType.ON_NOISE_LLR
    elif  (allocationTypeStr == 'incremental-pre'):
        allocationType = AllocationType.INCREMENTAL_PRE
    elif  (allocationTypeStr == 'incremental-post'):
        allocationType = AllocationType.INCREMENTAL_POST
    else:
        print("Non valid input for parameter [Features].allocationType. Valid values are: 'random', 'split', 'on-noise', 'on-noise-odds', 'on-noise-llr', 'incremental-pre', 'incremental-post'")
        sys.exit()
        
    ## -- Collective Decision params
    neighbourSize = config.getint('CollectiveDecision', 'neighbourSize')
    maxLoops = config.getint('CollectiveDecision', 'maxIterations')
    ## -- Network params
    numOfNodes = config.getint('Network', 'number_of_nodes')
    
    if DEBUG:
        print( "randomSeed: " + str(randomSeed) )
        print( "numberOfExperiments: " + str(numberOfExperiments) )
        print( "outputTxtFile: " + str(outputTxtFile) )
        print( "outputPdfFile: " + str(outputPdfFile) )
        print( "cluster: " + str(cluster) )
        print( "numFeatures: " + str(numFeatures) )
        print( "featuresAccuracies: " + str(featuresAccuracies) )
        print( "allocationType: " + str(allocationTypeStr) )
        print( "numOfNodes: " + str(numOfNodes) )
    
    ## Open output file and create the first line
    os.makedirs(os.path.dirname(outputTxtFile), exist_ok=True)
    if not cluster: os.makedirs(os.path.dirname(outputPdfFile), exist_ok=True)
    outFile = open(outputTxtFile, 'w')
    extraInfo = ''
    line = 'seed \t exp \t iter \t conv \t dec ' + extraInfo + '\n'
    outFile.write(line)
    
    count_correct  = 0
    count_wrong    = 0
    count_undecided= 0
    for exp in range(1,numberOfExperiments+1):
        seed = randomSeed+exp
        rand.seed(seed)
        
        ## init the Population object
        pop = Population.CatPopulation(numOfNodes, numFeatures, featuresAccuracies, allocationType, DEBUG)

        if (DEBUG): print("Initialising agents")
        pop.initPopulation(neighbourSize+1)
        pop.sampleFeatures()
        ### Now, the nodes interact with each other
        results = pop.collectiveDecision(neighbourSize, maxLoops)
        line = str(seed) + '\t' + str(exp) + '\t' + str(results[0]) + '\t' + str(results[1]) + '\t' + str(results[2]) + '\n'
        outFile.write(line)
        
        if min(results[1]) < 0.9999999999:
            count_undecided += 1
        elif max(results[2]) == 0:
            count_correct += 1
        else:
            count_wrong += 1
    
    print("correct,wrong,undecided: " + str([count_correct,count_wrong,count_undecided]))
    outFile = open(outputTxtFile, 'w')
    
    outFile.close()
    if DEBUG: print("Process Ended")
    