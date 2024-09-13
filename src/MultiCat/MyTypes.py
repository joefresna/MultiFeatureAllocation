'''
Created on 23 Aug 2018

@author: Andreagiovanni Reina.
University of Sheffield, UK.
'''
from enum import Enum

class AllocationType(Enum):
    RANDOM = 0
    SPLIT = 1
    ON_NOISE = 2
    ON_NOISE_ODDS = 3
    ON_NOISE_LLR = 4
    INCREMENTAL_PRE = 5
    INCREMENTAL_POST = 6
    ON_NOISE_ERR = 7
    ON_NOISE_VAR = 8
    
class SelfAllocationType(Enum):
    INCREMENTAL_PRE = 5
    INCREMENTAL_POST = 6


class NetworkType(Enum):
    FULLY_CONNECTED = 0
    ERSOS_RENYI = 1
    BARABASI_ALBERT = 2
    SPACE = 3
    
class SpreadAccuracies(Enum):
    EQUALLY_SPACED=0
    MEAN_AND_CONSTANT_DIFF=1

    